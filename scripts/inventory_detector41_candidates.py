"""Audit confidence and trailing three-sample track continuity on normal FIT only."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    cfg=json.loads(Path('configs/experiment40_representation.json').read_text())
    manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text());result=[]
    for scene in cfg['scenes']:
        for part in ['representation_train','representation_validation']:
            role_scores=defaultdict(list);sample_count=0;any_count=0;all_count=0;sequences={}
            for seq in manifest['subsplits'][scene][part]:
                with np.load(Path(cfg['source_features'][scene])/f'training_{seq}.npz') as f:d=dict(f)
                n=len(d['indices']);sample_count+=n;qualified=np.zeros(len(d['roles']),bool)
                for track in np.unique(d['tracks']):
                    ids=np.flatnonzero(d['tracks']==track);ids=ids[np.argsort(d['object_frames'][ids],kind='stable')]
                    for k in range(2,len(ids)):
                        last=ids[k-2:k+1]
                        if np.all(np.diff(d['object_frames'][last])==1) and np.all(d['confidence'][last]>=.5):qualified[ids[k]]=True
                qa=0;qb=0
                for step in range(n):
                    ids=np.flatnonzero(d['object_frames']==step)
                    qa+=int(qualified[ids].any());qb+=int(len(ids)>0 and qualified[ids].all())
                sequences[seq]={'any_qualified':qa,'all_cached_boxes_qualified':qb};any_count+=qa;all_count+=qb
                for role in np.unique(d['roles']):role_scores[str(int(role))].extend(d['confidence'][d['roles']==role].tolist())
            result.append({'scene':scene,'partition':part,'sampled_frames':sample_count,'any_high_conf_continuous_box_frames':any_count,'all_cached_boxes_high_conf_continuous_frames':all_count,
                           'roles':{k:{'count':len(v),'ge_05':int((np.array(v)>=.5).sum()),'quantiles':np.quantile(v,[0,.5,.9,1]).tolist()} for k,v in sorted(role_scores.items())},'per_sequence':sequences})
    # Preserve first audit, verify every support count independently (quantile definition is explicit here).
    previous=json.loads(Path('results/experiment41/initial_candidate_inventory.json').read_text())
    for a,b in zip(result,previous):
        for key in ['scene','partition','sampled_frames','any_high_conf_continuous_box_frames','all_cached_boxes_high_conf_continuous_frames','per_sequence']:assert a[key]==b[key],(a['scene'],key)
        for role in a['roles']:
            for key in ['count','ge_05']:assert a['roles'][role][key]==b['roles'][role][key]
    Path('results/experiment41/candidate_inventory_recomputed.json').write_text(json.dumps({'quantile_levels':[0,.5,.9,1],'original_support_counts_reproduced':True,'rows':result},indent=2)+'\n')
    print('Candidate support counts reproduced')


if __name__=='__main__':main()
