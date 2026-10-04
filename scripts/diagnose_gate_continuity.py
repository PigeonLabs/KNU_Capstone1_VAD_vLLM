"""Normal FIT-only gate sign changes and lost-relation run lengths."""
import json
from pathlib import Path
import numpy as np


def runs(mask):
    padded=np.r_[False,mask,False];change=np.flatnonzero(padded[1:]!=padded[:-1]);return list(zip(change[::2],change[1::2]))


def main():
    split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];rows=[];lengths=[]
    for seq in split['fit']:
        with np.load(f'artifacts/experiment16/features/R04/training_{seq}.npz') as old,np.load(f'artifacts/experiment17/features/R04/training_{seq}.npz') as new:
            lost=old['relation_valid']&~new['relation_valid'];widths=np.diff(np.r_[new['indices'],int(new['frame_count'])]);gaps=runs(lost)
            adjacent=flips=0
            for track in np.unique(new['tracks'][new['roles']==1]):
                ids=np.flatnonzero((new['roles']==1)&(new['tracks']==track));ids=ids[np.argsort(new['object_frames'][ids],kind='stable')]
                consecutive=np.diff(new['object_frames'][ids])==1;accepted=new['anchor_margin'][ids]>0
                adjacent+=int(consecutive.sum());flips+=int(np.sum(consecutive&(accepted[1:]!=accepted[:-1])))
            lengths.extend([int(end-start) for start,end in gaps]);rows.append({'sequence':seq,'consecutive_same_track_pairs':adjacent,'margin_sign_changes':flips,'lost_relation_samples':int(lost.sum()),'lost_relation_runs':len(gaps),'one_sample_lost_runs':int(sum(end-start==1 for start,end in gaps)),'lost_run_lengths_source_frames':[int(widths[start:end].sum()) for start,end in gaps]})
    result={'normal_fit_only':True,'consecutive_same_track_pairs':sum(r['consecutive_same_track_pairs'] for r in rows),'margin_sign_changes':sum(r['margin_sign_changes'] for r in rows),'lost_relation_samples':sum(r['lost_relation_samples'] for r in rows),'lost_relation_runs':len(lengths),'one_sample_lost_runs':sum(x==1 for x in lengths),'median_lost_run_samples':float(np.median(lengths)) if lengths else None,'sequences':rows,'limitations':['Sign flips are not ground-truth role errors.','Lost relations also depend on area gates, target availability and candidate selection.','Temporal smoothing could preserve persistent wrong anchors or delay a correct decision.']}
    Path('results/experiment17/normal_gate_continuity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sequences'},indent=2))


if __name__=='__main__':main()
