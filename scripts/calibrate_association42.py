"""Freeze association gates using reviewed normal calibration negatives only."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from ipad_vad.learned_association import ResidualMetric
from ipad_vad.learned_detector import sha,write
from train_association42 import load_pairs,OUT,ART

def main():
    if (OUT/'association_gates.json').exists():raise RuntimeError('Gates already frozen')
    torch.set_num_threads(4);cfg=json.loads(Path('configs/experiment42_association.json').read_text());tr=json.loads((OUT/'training.json').read_text());assert sha(tr['checkpoint'])==tr['checkpoint_sha256'];model=ResidualMetric(rank=cfg['rank']);model.load_state_dict(torch.load(tr['checkpoint'],map_location='cpu',weights_only=True));model.eval()
    a,b,neg,groups,rows=load_pairs('normal_calibration');a=a[neg];b=b[neg];rows=[r for r,n in zip(rows,neg) if n];settings=cfg['calibration'];review=json.loads((OUT/'negative_pair_review.json').read_text())['pairs'];result={}
    for branch in ['raw','learned']:
        with torch.no_grad():
            x,y=(torch.nn.functional.normalize(torch.from_numpy(z),dim=-1) for z in [a,b])
            if branch=='learned':x,y=model(x),model(y)
            similarity=(x*y).sum(-1).numpy()
        gates={};diagnostics=[]
        for scene in ['R01','R02','R03','R04']:
            for role in range(2 if scene=='R03' else 3):
                ix=[i for i,r in enumerate(rows) if r['scene']==scene and int(r['role'])==role];count=len(ix);videos=len({rows[i]['sequence'] for i in ix});support={p:sum(r['status']=='accept' and r['scene']==scene and int(r['role'])==role and r['partition']==p for r in review) for p in ['representation_train','representation_validation']};supported=count>=settings['minimum_negative_pairs'] and videos>=settings['minimum_negative_videos'] and all(support.values());threshold=float(similarity[ix].max()+settings['margin']) if supported else None;enabled=supported and threshold<=1;gates[f'{scene}/{role}']=threshold if enabled else None;diagnostics.append({'scene':scene,'role':role,'cal_negative_pairs':count,'cal_negative_videos':videos,'training_support':support,'gate':threshold,'enabled':enabled,'negative_cosine':similarity[ix].tolist()})
        result[branch]={'gates':gates,'diagnostics':diagnostics}
    write(OUT/'association_gates.json',{'normal_only':True,'checkpoint_sha256':tr['checkpoint_sha256'],'branches':result,'calibration_policy':settings,'source_sha256':{str(p):sha(p) for p in [Path(__file__),OUT/'training.json',OUT/'training_protocol.json',OUT/'negative_pair_review.json']}});print(json.dumps({b:r['gates'] for b,r in result.items()},indent=2))
if __name__=='__main__':main()
