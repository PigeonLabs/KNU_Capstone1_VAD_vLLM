"""Normal FIT-only comparison of instantaneous and causal-median gate continuity."""
import json
from pathlib import Path
import numpy as np


def lengths(mask):
    padded=np.r_[False,mask,False];cuts=np.flatnonzero(padded[1:]!=padded[:-1]);return (cuts[1::2]-cuts[::2]).tolist()


def main():
    split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];all_rows={};summary={}
    for experiment in ['17','18']:
        rows=[];gaps=[]
        for seq in split['fit']:
            with np.load(f'artifacts/experiment16/features/R04/training_{seq}.npz') as ungated,np.load(f'artifacts/experiment{experiment}/features/R04/training_{seq}.npz') as d:
                margin=d['anchor_gate_margin'] if experiment=='18' else d['anchor_margin'];adjacent=flips=0
                for track in np.unique(d['tracks'][d['roles']==1]):
                    ids=np.flatnonzero((d['roles']==1)&(d['tracks']==track));ids=ids[np.argsort(d['object_frames'][ids],kind='stable')];consecutive=np.diff(d['object_frames'][ids])==1
                    accepted=margin[ids]>0;adjacent+=int(consecutive.sum());flips+=int(np.sum(consecutive&(accepted[1:]!=accepted[:-1])))
                lost=ungated['relation_valid']&~d['relation_valid'];run_lengths=lengths(lost);gaps.extend(run_lengths)
                rows.append({'sequence':seq,'consecutive_same_track_pairs':adjacent,'sign_changes':flips,'lost_relation_samples_vs_ungated':int(lost.sum()),'lost_runs_vs_ungated':len(run_lengths),'one_sample_lost_runs_vs_ungated':sum(x==1 for x in run_lengths)})
        summary[experiment]={'consecutive_same_track_pairs':sum(r['consecutive_same_track_pairs'] for r in rows),'sign_changes':sum(r['sign_changes'] for r in rows),'lost_relation_samples_vs_ungated':sum(r['lost_relation_samples_vs_ungated'] for r in rows),'lost_runs_vs_ungated':len(gaps),'one_sample_lost_runs_vs_ungated':sum(x==1 for x in gaps),'median_lost_run_samples':float(np.median(gaps)) if gaps else None};all_rows[experiment]=rows
    result={'normal_fit_only':True,'comparators':summary,'sequences':all_rows,'note':'Relation-loss runs are measured against the same ungated experiment16 observations. Sign stability does not establish correct semantic identity; persistent wrong candidates can be stabilized.'}
    Path('results/experiment18/normal_temporal_continuity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
