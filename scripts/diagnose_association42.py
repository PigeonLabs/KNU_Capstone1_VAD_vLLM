"""Inspect eligible association opportunities without labels; render all added edges."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw
from ipad_vad.learned_detector import sha,write
from ipad_vad.learned_association import ResidualMetric,AssociationTracker
from ipad_vad.tracking import iou_matrix
from ipad_vad.data import frames_in_order
from experiment41_phases import load_npz
OUT=Path('results/experiment42');ART=Path('artifacts/experiment42')

def main():
    torch.set_num_threads(4);cfg=json.loads(Path('configs/experiment42_association.json').read_text());gates=json.loads((OUT/'association_gates.json').read_text());m=ResidualMetric();m.load_state_dict(torch.load(ART/'adapter/best.pt',map_location='cpu',weights_only=True));m.eval();root=Path(json.loads(Path('configs/experiment40_representation.json').read_text())['data_root']);counts=[];edge_reviews=[]
    for split in ['training','testing']:
        record_path=OUT/f'{split}_observations.json'
        if not record_path.exists():continue
        record=json.loads(record_path.read_text())
        for branch in ['raw','learned']:
            counter=Counter();similarity=[]
            for r in [r for r in record['sequences'] if r['branch']==branch and r['scene']=='R04']:
                d=load_npz(Path('artifacts/experiment41/detections/features/R04')/f'{split}_{r["sequence"]}.npz');tracker=AssociationTracker({0:gates['branches'][branch]['gates']['R04/0']},**cfg['association'])
                with torch.no_grad():
                    z=torch.nn.functional.normalize(torch.from_numpy(d['crop_features'].astype(np.float32)),dim=-1)
                    if branch=='learned':z=m(z)
                    z=z.numpy()
                for t in range(len(d['indices'])):
                    ix=np.flatnonzero(d['object_frames']==t);prior={k:v for k,v in tracker.active.items() if t-v['last']<=tracker.max_age and v['role']==0};current=ix[d['roles'][ix]==0];usedp=set();usedc=set()
                    if prior and len(current):
                        pp=list(prior);overlap=iou_matrix([prior[k]['box'] for k in pp],d['boxes'][current]);cost=np.where(overlap>=.2,1-overlap,1e6)
                        for i,j in zip(*linear_sum_assignment(cost)):
                            if overlap[i,j]>=.2:usedp.add(pp[i]);usedc.add(int(current[j]))
                    pp=[k for k,v in prior.items() if k not in usedp and v['confidence']>=.5];cc=[int(j) for j in current if j not in usedc and d['confidence'][j]>=.5]
                    if pp and cc:counter['unmatched_high_confidence_frames']+=1
                    for k in pp:
                        for j in cc:
                            counter['unmatched_high_confidence_pairs']+=1;pb=prior[k]['box'];cb=d['boxes'][j];dist=np.linalg.norm((pb[:2]+pb[2:]-cb[:2]-cb[2:])/2);ratio=np.maximum(pb[2:]-pb[:2],1e-9).prod()/np.maximum(cb[2:]-cb[:2],1e-9).prod()
                            if dist>.25 or ratio<.25 or ratio>4:counter['geometry_rejected_pairs']+=1;continue
                            counter['geometry_eligible_pairs']+=1;c=float(np.clip(prior[k]['feature']@z[j],-1,1));similarity.append(c)
                            if c<tracker.gates[0]:counter['cosine_rejected_pairs']+=1
                            else:counter['cosine_pass_pairs']+=1
                    tracker.update(d['boxes'][ix],d['roles'][ix],t,z[ix],d['confidence'][ix],ix)
                assert len(tracker.edges)==len(r['appearance_edges']);counter['accepted_appearance_edges']+=len(tracker.edges)
                for e in tracker.edges:
                    edge_reviews.append({'split':split,'scene':'R04','sequence':r['sequence'],'branch':branch,**e,'status':'unreviewed'})
            counts.append({'split':split,'branch':branch,**dict(counter),'geometry_eligible_cosines':similarity})
    dest=ART/'edge_review';dest.mkdir(exist_ok=True)
    for offset in range(0,len(edge_reviews),6):
        board=Image.new('RGB',(1536,864),'#151515');draw=ImageDraw.Draw(board)
        for cell,r in enumerate(edge_reviews[offset:offset+6]):
            x=(cell%2)*768;y=(cell//2)*288;d=load_npz(Path('artifacts/experiment41/detections/features/R04')/f'{r["split"]}_{r["sequence"]}.npz');images=frames_in_order(root/'R04'/r['split']/'frames'/r['sequence']);draw.text((x+2,y+4),f'{offset+cell} {r["split"]} R04/{r["sequence"]} {r["branch"]} {r["previous_detection"]}->{r["detection"]}',fill='white');r['image_sha256']={}
            for j,di in enumerate([r['previous_detection'],r['detection']]):
                fi=int(d['indices'][d['object_frames'][di]]);p=images[fi];im=Image.open(p).convert('RGB').resize((384,256));board.paste(im,(x+j*384,y+32));b=d['boxes'][di];draw.rectangle([x+j*384+b[0]*384,y+32+b[1]*256,x+j*384+b[2]*384,y+32+b[3]*256],outline='cyan',width=3);r['image_sha256'][str(p.relative_to(root))]=sha(p)
            r['board']=f'edges_{offset//6:02}.jpg'
        board.save(dest/f'edges_{offset//6:02}.jpg',quality=95)
    write(OUT/'association_diagnostics.json',{'counts':counts,'edge_review_drafts':edge_reviews,'boards_sha256':{str(p):sha(p) for p in dest.glob('*.jpg')},'no_labels_used':True,'interpretation':'Opportunity counts under branch-specific histories; track-number changes include later renumbering, not separate links. No independent IDF1/HOTA ground truth.'});print(json.dumps([{k:v for k,v in r.items() if k!='geometry_eligible_cosines'} for r in counts],indent=2))
if __name__=='__main__':main()
