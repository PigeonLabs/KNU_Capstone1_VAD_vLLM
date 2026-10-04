"""Post-result historical-rank and existing-route contrasts; no counterfactual fit."""
import json
from pathlib import Path
import numpy as np
from ipad_vad.data import hold_scores
from ipad_vad.missing_age import causal_age
from experiment37_asinh_phase import OUT,SOURCE,root,load,write


def main():
    diag=json.loads((OUT/'diagnostic.json').read_text());history={};ranks=json.loads((OUT/'fit_rank_control.json').read_text());cells={str(i):{'frames':0,'visual_changed_frames':0,'combined_changed_frames':0,'hold_only_fp':0,'hold_only_tp':0,'age_only_fp':0,'age_only_tp':0} for i in range(4)}
    unchanged=['labels','indices','phases','boxes','tracks','detected_roles','detected_object_frames','transition','transition_raw','transition_valid','transition_gated','dwell','dwell_valid','dwell_age','dwell_reason','dwell_entry_context','process']
    for gate in ['hold','pool','age']:
        old=json.loads(Path(f'results/experiment36_refit_{gate}/metrics.json').read_text());new=diag['variants'][f'37_control_{gate}'];changes=0;files=0
        for p in sorted(Path(f'artifacts/experiment37_control_{gate}/predictions').glob('*.npz')):
            a=load(p);b=load(Path(f'artifacts/experiment36_refit_{gate}/predictions')/p.name)
            for k in unchanged:np.testing.assert_array_equal(a[k],b[k])
            changes+=int(np.sum(a['combined']!=b['combined']));files+=1
        history[gate]={'previous_experiment36_refit_combined':old['metrics']['combined'],'matched_rank_control_combined':new['metrics']['combined'],'previous_fp':int(round(old['test_normal_frame_alarm_rate']*3576)),'previous_tp':int(round(old['test_anomaly_frame_recall_at_q99']*4578)),'current_fp':new['alarms']['normal'],'current_tp':new['alarms']['anomaly'],'prediction_files_checked':files,'unchanged_arrays':unchanged,'changed_combined_frames':changes}
    for p in sorted(SOURCE.glob('testing_*.npz')):
        seq=p.stem.split('_')[1];d=load(root('asinh')/p.name);age,_=causal_age(d);state=np.where(d['relation_valid'],0,np.where(age<0,1,np.where(age<=56,2,3)));dense=hold_scores(d['indices'],state,int(d['frame_count']));a=load(f'artifacts/experiment37_asinh_hold/predictions/R04_{seq}.npz');b=load(f'artifacts/experiment37_asinh_age/predictions/R04_{seq}.npz');qa=diag['variants']['37_asinh_hold']['q99'];qb=diag['variants']['37_asinh_age']['q99'];y=a['labels'];ha=a['combined']>qa;hb=b['combined']>qb
        for i in range(4):
            mask=dense==i;v=cells[str(i)];v['frames']+=int(mask.sum());v['visual_changed_frames']+=int(np.sum(mask&(a['visual']!=b['visual'])));v['combined_changed_frames']+=int(np.sum(mask&(a['combined']!=b['combined'])))
            for field,alarm,label in [('hold_only_fp',ha&~hb,0),('hold_only_tp',ha&~hb,1),('age_only_fp',hb&~ha,0),('age_only_tp',hb&~ha,1)]:v[field]+=int(np.sum(mask&alarm&(y==label)))
    write(OUT/'historical_rank_and_fallback_contrasts.json',{'historical_rank_comparison':history,'rank_changes':{k:{'previous':v,'matched_control':ranks['rank_maps']['control'][k]} for k,v in ranks['previous_rank_limits'].items() if v!=ranks['rank_maps']['control'][k]},'existing_asinh_hold_vs_age_by_state':cells,'state_names':{'0':'observed','1':'no_prior_observation','2':'missing_within_tau','3':'stale_missing'},'note':'Existing measured runs only, no new model selection. Previous same-phase control differs in PCA ranks/CDF. Hold/age comparison shares phase model/ranks and full-normal request CDFs but routing differs; a future initial-only gate still requires its own normal q99 validation.'})
    print(json.dumps({'history':history,'routing':cells},indent=2))

if __name__=='__main__':main()
