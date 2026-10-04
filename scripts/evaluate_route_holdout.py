"""Normal calibration-video holdout for role-wide versus bank-route CDFs."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from ipad_vad.route_calibration import leave_one_video_out
from ipad_vad.data import hold_scores


def load_cache(root,seq,part='training'):
    with np.load(root/f'{part}_{seq}.npz',allow_pickle=False) as f:d=dict(f)
    d['sequence_id']=f'R04/{part}_{seq}'
    return d


def normal_model_arrays(model):
    arrays={'threshold':np.array(model.threshold),'process_reference':model.process_reference}
    if model.missing_age is not None:
        arrays['appearance_age_tau']=np.array(model.missing_age.tau);arrays['appearance_age_fit_durations']=model.missing_age.durations
    for role,ref in model.calibration.items():arrays[f'calibration_{role}']=ref
    for state,ref in model.state_process_references.items():arrays[f'process_reference_state_{state}']=ref
    if model.route_calibration is not None:
        for (role,route),ref in model.route_calibration.references.items():arrays[f'route_calibration_{role}_{route}']=ref
    return arrays


def support_rows(model):
    return [] if model.route_calibration is None else [{'role':r,'route':'phase' if k else 'pooled',**v} for (r,k),v in sorted(model.route_calibration.support.items())]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiments',nargs='+',default=['19','20']);parser.add_argument('--output-experiment',default='20');parser.add_argument('--feature-source-experiment',default='20');args=parser.parse_args();variants=args.experiments
    out=Path(f'results/experiment{args.output_experiment}');out.mkdir(parents=True,exist_ok=True);art=Path(f'artifacts/experiment{args.output_experiment}/normal_holdout');art.mkdir(parents=True,exist_ok=True)
    split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];root=Path(f'artifacts/experiment{args.feature_source_experiment}/features/R04');fit=[load_cache(root,s) for s in split['fit']];cal={s:load_cache(root,s) for s in split['calibration']}
    configs={e:json.loads(Path(f'configs/experiment{e}.json').read_text()) for e in variants};models={};rows=[]
    with threadpool_limits(limits=4):
        for e,cfg in configs.items():
            model=LognormalDwellBaseline(cfg,load_process(cfg));model.fit(fit);models[e]=model
        by_fit_mode={}
        for e,model in models.items():
            mode=configs[e].get('appearance_fit_conditioning',configs[e].get('appearance_conditioning','phase'))
            mode=(mode,json.dumps(configs[e].get('appearance_fit_sampling'),sort_keys=True),json.dumps(configs[e].get('appearance_phase_ranks'),sort_keys=True))
            if mode in by_fit_mode:
                other=by_fit_mode[mode];assert set(model.spaces)==set(other.spaces)
                for key in model.spaces:
                    for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(model.spaces[key],attr),getattr(other.spaces[key],attr))
            else:by_fit_mode[mode]=model
        for held,used in leave_one_video_out(split['calibration']):
            scores_by_variant={};d=cal[held];n=int(d['frame_count'])
            for e in variants:
                model=copy.deepcopy(models[e]);model.calibrate([cal[s] for s in used]);r=model.score(d);scores_by_variant[e]=r
                refs=normal_model_arrays(model);np.savez_compressed(art/f'{e}_exclude_{held}_references.npz',**refs)
                saved={k:r[k] for k in ['visual','transition','dwell','process','combined','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']};saved.update(indices=d['indices'],relation_valid=d['relation_valid'],global_route=model.appearance_routes(d,-1,np.arange(len(d['indices']))))
                roles=[];frames=[];routes=[];values=[];conditional=[]
                for role,ids,s in r['objects']:
                    route=model.appearance_routes(d,role,ids);roles.extend([role]*len(ids));frames.extend(ids);routes.extend(route);values.extend(s)
                    conditional.extend([model.route_calibration is not None and (role,int(k)) in model.route_calibration.references for k in route])
                saved.update(object_roles=np.array(roles),object_frames=np.array(frames),object_routes=np.array(routes),object_scores=np.array(values),object_route_calibrated=np.array(conditional))
                np.savez_compressed(art/f'{e}_exclude_{held}_scores.npz',**saved)
                alarm=r['combined']>model.threshold;dense=hold_scores(d['indices'],alarm,n).astype(bool);valid=hold_scores(d['indices'],d['relation_valid'],n).astype(bool);route=hold_scores(d['indices'],saved['global_route'],n)
                cal_scores=[model.score(cal[s]) for s in used];branches=['visual','transition','dwell','process','combined']
                finite=all(np.isfinite(x[k]).all() for x in cal_scores for k in branches);bounded=all(np.all((x[k]>=0)&(x[k]<=1)) for x in cal_scores for k in branches)
                row={'variant':e,'held_out_sequence':held,'calibration_sequences':used,'normal_q99':model.threshold,'calibration_finite':finite,'calibration_unit_interval':bounded,'calibration_combined_at_one':int(sum(np.sum(x['combined']==1) for x in cal_scores)),'held_out_samples':len(alarm),'held_out_sample_alarms':int(alarm.sum()),'held_out_frames':n,'held_out_frame_alarms':int(dense.sum()),'held_out_at_one_samples':int(np.sum(r['combined']==1)),'relation_strata':{},'global_bank_strata':{},'object_bank_strata':[],'route_support':support_rows(model)}
                for observed in [False,True]:
                    mask=valid==observed;row['relation_strata'][str(observed)]={'frames':int(mask.sum()),'alarms':int(np.sum(dense&mask))}
                for kind in [0,1]:
                    mask=route==kind;row['global_bank_strata']['phase' if kind else 'pooled']={'frames':int(mask.sum()),'alarms':int(np.sum(dense&mask))}
                for role in np.unique(saved['object_roles']):
                    for kind in [0,1]:
                        mask=(saved['object_roles']==role)&(saved['object_routes']==kind)
                        row['object_bank_strata'].append({'role':int(role),'route':'phase' if kind else 'pooled','observations':int(mask.sum()),'alarms':int(np.sum(mask&(saved['object_scores']>model.threshold))),'conditional_cdf_observations':int(np.sum(mask&saved['object_route_calibrated']))})
                assert f'R04/training_{held}' not in {s for v in row['route_support'] for s in v['videos']}
                rows.append(row)
            for e in variants[1:]:
                for key in ['transition','dwell','process','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']:np.testing.assert_array_equal(scores_by_variant[variants[0]][key],scores_by_variant[e][key])
    totals={}
    for e in variants:
        items=[r for r in rows if r['variant']==e];total={k:sum(r[k] for r in items) for k in ['held_out_samples','held_out_sample_alarms','held_out_frames','held_out_frame_alarms','held_out_at_one_samples']}
        total['frame_alarm_rate']=total['held_out_frame_alarms']/total['held_out_frames'];total['q99_range']=[min(r['normal_q99'] for r in items),max(r['normal_q99'] for r in items)];total['unsupported_routes']=sum(not r['supported'] for item in items for r in item['route_support']);totals[e]=total
    result={'normal_only':True,'fit_sequences':split['fit'],'holdout_sequences':split['calibration'],'totals':totals,'folds':rows,'note':'Each held-out normal video excluded from appearance/process references and q99. Fixed normal FIT models; no parameter selection. Global bank strata assign the combined frame alarm by the global branch route; object bank strata are individual feature observations, not unique frames.'}
    (out/'normal_holdout.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
