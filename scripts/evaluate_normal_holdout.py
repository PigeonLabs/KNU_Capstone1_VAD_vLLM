"""Four normal-video holdouts; never reads testing features or anomaly labels."""
import csv,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.experiment import load_process
from ipad_vad.data import hold_scores
from ipad_vad.context_dwell import observed_entry_context
from ipad_vad.normal_validation import leave_one_video_out,make_model,fit_arrays,array_fingerprint,reference_arrays,calibrate_holdout


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def scores_array(results,ids,data):
    keys=['visual','process','combined']
    if 'dwell' in results[0]:keys+=['transition','dwell','dwell_valid','dwell_age','dwell_reason']
    return {**{key:np.concatenate([r[key] for r in results]) for key in keys},
            'sequence':np.concatenate([np.full(len(data[s]['indices']),s) for s in ids]),
            'indices':np.concatenate([data[s]['indices'] for s in ids]),'phases':np.concatenate([data[s]['phases'] for s in ids])}


def main():
    cfg=json.loads(Path('configs/experiment13.json').read_text());scene=cfg['scene'];split=json.loads(Path(cfg['split_source']).read_text())[scene]
    folds=leave_one_video_out(split['fit'],split['calibration']);root=Path(f'artifacts/experiment{cfg["source_features_experiment"]}/features/{scene}')
    caches={s:load(root/f'training_{s}.npz') for s in split['fit']+split['calibration']};rows=[];detail=[];fit_checks={}
    with threadpool_limits(limits=4):
        for variant in cfg['variants']:
            vc=json.loads(Path(f'configs/experiment{variant}.json').read_text());assert vc['scene']==scene and vc['seed']==cfg['seed']
            fitted=make_model(vc,load_process(vc));fitted.fit([caches[s] for s in split['fit']]);arrays=fit_arrays(fitted)
            original=load(Path(f'artifacts/experiment{variant}/normal_model.npz'))
            for key,value in arrays.items():
                if key!='allowed':np.testing.assert_array_equal(value,original[key],err_msg=variant+':'+key)
            fingerprint=array_fingerprint(arrays);fit_checks[variant]={'fit_arrays_match_original':True,'fit_fingerprint':fingerprint}
            for fold in folds:
                held=fold['held_out'];cal_ids=fold['calibration'];model,cal_results,result=calibrate_holdout(fitted,caches,cal_ids,held)
                d=caches[held];cal=scores_array(cal_results,cal_ids,caches);saved=scores_array([result],[held],caches)
                saved['frame_count']=d['frame_count'];saved['process_conditioned']=model.conditioning_mask(d)
                saved['entry_context']=observed_entry_context(d)
                for branch in ('visual','process','combined'):saved['dense_'+branch]=hold_scores(d['indices'],result[branch],int(d['frame_count']))
                alarm=saved['dense_combined']>model.threshold;saved['dense_alarm']=alarm
                dest=Path(f'artifacts/experiment13/variant{variant}/holdout_{held}');dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/'normal_model.npz',**fit_arrays(model),**reference_arrays(model))
                np.savez_compressed(dest/'calibration_scores.npz',**cal);np.savez_compressed(dest/'heldout_scores.npz',**saved)
                entry=hold_scores(d['indices'],saved['entry_context'],len(alarm));phase=hold_scores(d['indices'],d['phases'],len(alarm));contexts={}
                dwell_dense=hold_scores(d['indices'],result['dwell'],len(alarm)) if 'dwell' in result else None
                valid_dense=hold_scores(d['indices'],result['dwell_valid'],len(alarm)) if 'dwell' in result else None
                for a,b in np.unique(np.column_stack([entry,phase]),axis=0):
                    use=(entry==a)&(phase==b);contexts[f'{a}->{b}']={'frames':int(use.sum()),'alarm_frames':int((alarm&use).sum()),'visual_at_one':int((use&(saved['dense_visual']==1)).sum()),
                        'dwell_valid_frames':int((use&valid_dense).sum()) if valid_dense is not None else None,
                        'dwell_at_one_valid':int((use&valid_dense&(dwell_dense==1)).sum()) if valid_dense is not None else None}
                row={'variant':variant,'held_out':held,'calibration_ids':','.join(cal_ids),'frames':len(alarm),'false_positive_frames':int(alarm.sum()),'frame_fpr':float(alarm.mean()),
                    'samples':len(d['indices']),'sample_fpr':float(np.mean(result['combined']>model.threshold)),'q99':model.threshold,
                    'calibration_samples':len(cal['combined']),'calibration_sample_fpr':float(np.mean(cal['combined']>model.threshold)),
                    'process_fallback_samples':int((~saved['process_conditioned']).sum()),'process_insufficient_support_fallback_samples':int((~saved['process_conditioned'][1:]).sum()),
                    'dwell_reference_samples':len(model.dwell.reference) if hasattr(model,'dwell') else None,
                    'dwell_valid_frames':int(valid_dense.sum()) if valid_dense is not None else None}
                evidence={'variant':variant,'held_out':held,'calibration_ids':cal_ids,'fit_ids':split['fit'],'fit_fingerprint':fingerprint,
                    'visual_reference_samples':{str(role):len(ref) for role,ref in model.calibration.items()},
                    'process_reference_samples':len(model.process_reference),'previous_state_reference_samples':{str(state):len(ref) for state,ref in model.state_process_references.items()},
                    'contexts':contexts,'heldout_visual_at_one':int((saved['dense_visual']==1).sum()),'heldout_process_at_one':int((saved['dense_process']==1).sum())}
                if hasattr(model,'dwell'):
                    evidence['dwell_fit_support']=model.dwell.support
                    evidence['dwell_reason_counts_sampled']={str(i):int((result['dwell_reason']==i).sum()) for i in range(4)}
                    evidence['dwell_raw_heldout_quantiles_valid']=np.quantile(model.dwell.raw(d)[0][result['dwell_valid']],[0,.5,.99,1]).tolist() if result['dwell_valid'].any() else None
                    evidence['dwell_reference_quantiles']=np.quantile(model.dwell.reference,[0,.5,.99,1]).tolist()
                    if hasattr(model.dwell,'context_support'):evidence['dwell_context_support']=model.dwell.context_support
                rows.append(row);detail.append(evidence);print(json.dumps(row),flush=True)
    summaries={}
    for variant in cfg['variants']:
        selected=[r for r in rows if r['variant']==variant];total=sum(r['frames'] for r in selected);fp=sum(r['false_positive_frames'] for r in selected)
        summaries[variant]={'heldout_videos':len(selected),'frames':total,'false_positive_frames':fp,'micro_frame_fpr':fp/total,
            'macro_video_frame_fpr':float(np.mean([r['frame_fpr'] for r in selected])),
            'video_frame_fpr_min_max':[min(r['frame_fpr'] for r in selected),max(r['frame_fpr'] for r in selected)],
            'q99_min_max':[min(r['q99'] for r in selected),max(r['q99'] for r in selected)]}
    out=Path('results/experiment13')
    with (out/'fold_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    result={'scene':scene,'protocol':cfg['protocol'],'fit_sequences':split['fit'],'heldout_pool':split['calibration'],'folds':rows,'summaries':summaries,'fit_checks':fit_checks,
            'not_applicable':['Anomaly AUROC/AP','Anomaly recall','Anomaly event latency'],
            'limitations':['Only four normal calibration videos.','Original recording-group independence unknown.','Fixed R03 development representation; not full IPAD or a final independent anomaly test.','Calibration video scores used to set q99; heldout video is excluded from every score reference.']}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');(out/'fold_diagnostics.json').write_text(json.dumps(detail,indent=2)+'\n')

if __name__=='__main__':main()
