"""Persist Codex visual negative-pair review and prepare a weak-positive audit."""
import csv,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageDraw
from ipad_vad.learned_detector import sha,write
from ipad_vad.data import frames_in_order
from ipad_vad.tracking import iou_matrix
OUT=Path('results/experiment42');ART=Path('artifacts/experiment42/pair_review')
# Zero-based rows of immutable drafts; all nine contact sheets inspected.
ACCEPT={1,4,5,7,8,9,10,13,14,15,16,17,18,19,22,25,26,28,29,31,34,37,38,40,41,43,46,47,49,52,55,58,59,60,61,62,64,65,66,67,70,71}
def main():
    drafts=json.loads((OUT/'negative_pair_review_drafts.json').read_text());inventory=json.loads((OUT/'pair_inventory.json').read_text());assert sha(OUT/'negative_pair_review_drafts.json')==inventory['negative_drafts_sha256'];assert len(drafts)==73
    for i,r in enumerate(drafts):
        r['status']='accept' if i in ACCEPT else 'reject';r['reviewer']='Codex visual inspection; not human identity ground truth';r['reason']='Two separately visible single material pieces in the same frame.' if i in ACCEPT else ('Role confusion: meter and scissor detections share a role label.' if i==0 else 'At least one box aggregates multiple/offset/stacked pieces, or single physical identity is ambiguous.');r['draft_index']=i
    write(OUT/'negative_pair_review.json',{'drafts_sha256':inventory['negative_drafts_sha256'],'reviewed_boards':inventory['boards'],'policy':'Accept only separately visible individual pieces; corrugated thickness alone is not multiple identity. Reject aggregate boxes and ambiguous identity. All 73 selected pairs reviewed before learning.','counts':dict(Counter(r['partition']+'/'+r['status'] for r in drafts)),'pairs':drafts})
    rows=list(csv.DictReader((OUT/'pair_candidates.csv').open()));groups=defaultdict(list);cache={}
    for r in rows:
        if r['kind']!='positive_candidate' or r['partition']=='normal_calibration':continue
        key=(r['scene'],r['sequence'])
        if key not in cache:
            with np.load(Path('artifacts/experiment41/detections/features')/key[0]/f'training_{key[1]}.npz') as f:cache[key]=dict(f)
        d=cache[key];a,b=int(r['detection_a']),int(r['detection_b']);r['minimum_confidence']=float(d['confidence'][[a,b]].min());r['iou']=float(iou_matrix(d['boxes'][[a]],d['boxes'][[b]])[0,0]);groups[(r['scene'],r['role'],r['partition'])].append(r)
    selected=[]
    for key,rs in sorted(groups.items()):
        for criterion in ['minimum_confidence','iou']:
            candidate=sorted(rs,key=lambda r:(r[criterion],r['id']))[0]
            if candidate['id'] not in {r['id'] for r in selected}:selected.append({**candidate,'selection':criterion})
    root=Path(json.loads(Path('configs/experiment40_representation.json').read_text())['data_root'])
    for offset in range(0,len(selected),8):
        board=Image.new('RGB',(1536,1152),'#151515');draw=ImageDraw.Draw(board)
        for cell,r in enumerate(selected[offset:offset+8]):
            x=(cell%2)*768;y=(cell//2)*288;d=cache[(r['scene'],r['sequence'])];images=frames_in_order(root/r['scene']/'training/frames'/r['sequence']);r['image_sha256']={};r['boxes']=[]
            draw.text((x+2,y+3),f'{offset+cell} '+r['id']+' '+r['partition'].removeprefix('representation_'),fill='white')
            for j,(di,fi) in enumerate([(int(r['detection_a']),int(r['frame_a'])),(int(r['detection_b']),int(r['frame_b']))]):
                p=images[fi];r['image_sha256'][str(p.relative_to(root))]=sha(p);box=d['boxes'][di];r['boxes'].append(box.tolist());im=Image.open(p).convert('RGB').resize((384,256));board.paste(im,(x+j*384,y+32));draw.rectangle((x+j*384+box[0]*384,y+32+box[1]*256,x+j*384+box[2]*384,y+32+box[3]*256),outline='#2de0dc',width=3)
            r['board']=f'positive_{offset//8:02}.jpg'
        board.save(ART/f'positive_{offset//8:02}.jpg',quality=94)
    write(OUT/'positive_pair_review_drafts.json',selected)
    print('Negatives:',dict(Counter(r['partition']+'/'+r['status'] for r in drafts)),'positive audit:',len(selected))
if __name__=='__main__':main()
