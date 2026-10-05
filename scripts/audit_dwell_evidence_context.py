"""Explain the fixed dwell gate and audit normal duration support without refitting."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from ipad_vad.context_dwell import complete_context_runs
from ipad_vad.dwell_episodes import extract_dwell_episodes
from experiment39_dwell_evidence import OUT,SOURCE,SPLIT,GATES,load,write,scalar_evidence


def main():
    normal=[];legacy=[];strict=[];prefix=0
    for part in ['fit','calibration']:
        for seq in SPLIT[part]:
            d=load(SOURCE/f'training_{seq}.npz');e=extract_dwell_episodes(d);mask,_=scalar_evidence(d)
            np.testing.assert_array_equal(mask,e['sample_age_frames']>=0)
            for row in e['episodes']:
                if row['entry_observed']:normal.append(dict(partition=part,sequence=seq,**row))
            legacy.extend({'partition':part,'sequence':seq,'context':f'{a}->{b}','duration':v} for (a,b),v in complete_context_runs(d))
            strict.extend({'partition':part,'sequence':seq,'context':f'{r["entry_context"]}->{r["phase"]}','duration':r['duration_frames']} for r in e['episodes'] if r['status']=='complete')
    summaries={}
    for part in ['fit','calibration']:
        contexts=sorted({r['context'] for r in legacy if r['partition']==part}|{f'{r["entry_context"]}->{r["phase"]}' for r in normal if r['partition']==part})
        summaries[part]={}
        for context in contexts:
            a=[r for r in legacy if r['partition']==part and r['context']==context];b=[r for r in strict if r['partition']==part and r['context']==context];c=[r for r in normal if r['partition']==part and f'{r["entry_context"]}->{r["phase"]}'==context and r['status']=='right_censored']
            summaries[part][context]={'legacy_complete':len(a),'same_pair_complete':len(b),'same_pair_complete_videos':sorted({r['sequence'] for r in b}),'same_pair_right_censored':len(c),'right_censored_with_positive_lower_bound':sum(r['censor_lower_bound_frames']>0 for r in c),'right_censor_causes':dict(Counter(r['end_reason'] for r in c)),'strict_meets_minimum10':len(b)>=10,'strict_complete_durations':sorted(r['duration'] for r in b)}
    changes=[]
    for p in sorted(SOURCE.glob('testing_*.npz')):
        seq=p.stem.split('_')[1];d=load(p);episode=extract_dwell_episodes(d);mask,_=scalar_evidence(d);np.testing.assert_array_equal(mask,episode['sample_age_frames']>=0)
        for gate in GATES:
            a=load(f'artifacts/experiment39_control_{gate}/predictions/R04_{seq}.npz');b=load(f'artifacts/experiment39_gated_{gate}/predictions/R04_{seq}.npz');y=a['labels'];blocked=a['dwell_valid']&~b['dwell_evidence_valid']
            row={'gate':gate,'sequence':seq,'blocked_frames':int(blocked.sum())}
            for key in ['dwell_gated','process','combined']:
                change=a[key]!=b[key];assert not np.any(change&~blocked)
                row[key+'_changed_normal']=int(np.sum(change&(y==0)));row[key+'_changed_anomaly']=int(np.sum(change&(y==1)))
            changes.append(row)
    summary={g:{k:sum(r[k] for r in changes if r['gate']==g) for k in changes[0] if k not in ['gate','sequence']} for g in GATES}
    write(OUT/'duration_and_gate_context.json',{'normal_only_duration_fit_audit':True,'no_distribution_refit':True,'normal_duration_support':summaries,'normal_observed_entry_episodes':normal,'normal_legacy_complete':legacy,'normal_strict_complete':strict,'test_gate_effect_per_sequence':changes,'test_gate_effect_summary':summary,'episode_parser_mask_equality_videos':44,'note':'Complete and right-censored observations remain separate. No missing interval is added as exposure. Censoring may depend on process/occlusion; no survival estimator or support threshold was selected.'})
    print(json.dumps({'normal_support':summaries,'test_changes':summary},indent=2),flush=True)

if __name__=='__main__':main()
