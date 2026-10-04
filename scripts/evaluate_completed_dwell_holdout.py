"""Experiment14 normal holdout with completed-run percentile, before development test."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.completed_dwell import CompletedDwellBaseline
from ipad_vad.normal_validation import fit_arrays,reference_arrays,calibrate_holdout,array_fingerprint,leave_one_video_out
from ipad_vad.experiment import load_process
from ipad_vad.data import hold_scores


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    cfg=json.loads(Path('configs/experiment14.json').read_text());scene=cfg['scene'];split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    source=Path(f'artifacts/experiment14/features/{scene}');data={s:load(source/f'training_{s}.npz') for s in split['fit']+split['calibration']};rows=[]
    with threadpool_limits(limits=4):
        fitted=CompletedDwellBaseline(cfg,load_process(cfg));fitted.fit([data[s] for s in split['fit']]);fp=array_fingerprint(fit_arrays(fitted))
        original=json.loads(Path('results/experiment13/metrics.json').read_text());assert fp==original['fit_checks']['12']['fit_fingerprint']
        for fold in leave_one_video_out(split['fit'],split['calibration']):
            held=fold['held_out'];ids=fold['calibration'];model,cal,result=calibrate_holdout(fitted,data,ids,held);d=data[held]
            oldroot=Path(f'artifacts/experiment13/variant12/holdout_{held}');oldmodel=load(oldroot/'normal_model.npz');old=load(oldroot/'heldout_scores.npz')
            arrays={**fit_arrays(model),**reference_arrays(model)}
            for key,value in arrays.items():
                if key not in ('dwell_reference','threshold'):np.testing.assert_array_equal(value,oldmodel[key])
            for key in ('visual','transition','dwell_valid','dwell_age','dwell_entry_context'):
                expected=old['entry_context'] if key=='dwell_entry_context' else old[key];np.testing.assert_array_equal(result[key],expected)
            scores={key:result[key] for key in ['visual','process','combined','transition','dwell','dwell_valid','dwell_age','dwell_entry_context']}
            scores.update(indices=d['indices'],phases=d['phases'],frame_count=d['frame_count'])
            scores['dense_combined']=hold_scores(d['indices'],result['combined'],int(d['frame_count']));alarm=scores['dense_combined']>model.threshold;scores['dense_alarm']=alarm
            cal_scores={key:np.concatenate([r[key] for r in cal]) for key in ['visual','process','combined','transition','dwell','dwell_valid']}
            assert model.threshold==np.quantile(cal_scores['combined'],cfg['calibration_quantile'],method='higher') and len(model.dwell.reference)==0
            dest=Path(f'artifacts/experiment14/normal_holdout/holdout_{held}');dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'normal_model.npz',**arrays);np.savez_compressed(dest/'heldout_scores.npz',**scores);np.savez_compressed(dest/'calibration_scores.npz',**cal_scores)
            row={'held_out':held,'calibration_ids':ids,'frames':len(alarm),'false_positive_frames':int(alarm.sum()),'frame_fpr':float(alarm.mean()),
                 'sampled_fpr':float(np.mean(result['combined']>model.threshold)),'q99':model.threshold,
                 'baseline_false_positive_frames':int(old['dense_alarm'].sum()),'baseline_frame_fpr':float(old['dense_alarm'].mean()),'baseline_q99':float(oldmodel['threshold']),
                 'added_alarms':int((alarm&~old['dense_alarm']).sum()),'removed_alarms':int((~alarm&old['dense_alarm']).sum()),
                 'fit_fingerprint':fp,'dwell_reference_source':'FIT complete context durations; no calibration-age CDF'}
            rows.append(row);print(json.dumps(row),flush=True)
    total=sum(r['frames'] for r in rows);alarms=sum(r['false_positive_frames'] for r in rows)
    out={'scene':scene,'folds':rows,'frames':total,'false_positive_frames':alarms,'micro_frame_fpr':alarms/total,'macro_video_frame_fpr':float(np.mean([r['frame_fpr'] for r in rows])),
         'existing_fit_models_references_and_visual_transition_preserved':True,'same_dwell_valid_masks':True,
         'limitations':'Same four normal development videos used to motivate this change. This is a regression check, not a newly untouched independent validation.'}
    Path('results/experiment14/normal_holdout.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
