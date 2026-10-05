"""Inventory normal weak temporal pairs without treating tracker IDs as labels."""
import csv,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image,ImageDraw
from ipad_vad.tracking import iou_matrix
from ipad_vad.data import frames_in_order
from ipad_vad.learned_detector import sha,write

OUT=Path('results/experiment42');ART=Path('artifacts/experiment42')

def main():
    if (OUT/'pair_inventory.json').exists():raise RuntimeError('Pair inventory exists; preserve review provenance')
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            s=str(args[0])
            if '/test_label/' in s or '/testing/' in s or Path(s).name.startswith('testing_'):raise RuntimeError('Normal pairs only')
    sys.addaudithook(guard)
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path('configs/experiment40_representation.json').read_text());split=json.loads(Path('results/experiment40/data_manifest.json').read_text())['subsplits'];source=json.loads(Path('results/experiment41/detector_training_extraction.json').read_text());hashes={(r['scene'],r['sequence']):r['sha256'] for r in source['sequences']};rows=[];negative_review=[];source_files={};counts=[]
    for scene in cfg['scenes']:
        for group in ['representation_train','representation_validation','normal_calibration']:
            for seq in split[scene][group]:
                path=Path('artifacts/experiment41/detections/features')/scene/f'training_{seq}.npz';assert sha(path)==hashes[(scene,seq)];source_files[str(path)]=hashes[(scene,seq)]
                with np.load(path,allow_pickle=False) as f:d=dict(f)
                images=frames_in_order(Path(cfg['data_root'])/scene/'training/frames'/seq)
                for role in range(2 if scene=='R03' else 3):
                    positives=[];negatives=[];ids_by_frame=[np.flatnonzero((d['roles']==role)&(d['object_frames']==t)&(d['confidence']>=.5)) for t in range(len(d['indices']))]
                    def pair(a,b,kind):
                        t0,t1=map(int,d['object_frames'][[a,b]])
                        return {'id':f'{scene}_{seq}_{role}_{a}_{b}_{kind}','scene':scene,'sequence':seq,'partition':group,'role':role,'kind':kind,'detection_a':int(a),'detection_b':int(b),'sample_a':t0,'sample_b':t1,'frame_a':int(d['indices'][t0]),'frame_b':int(d['indices'][t1])}
                    for t,ids in enumerate(ids_by_frame):
                        if len(ids)>1:
                            a,b=np.where(np.triu(iou_matrix(d['boxes'][ids],d['boxes'][ids])<=.1,1))
                            negatives.extend(pair(ids[i],ids[j],'negative_candidate') for i,j in zip(a,b))
                        if not t or not len(ids) or not len(ids_by_frame[t-1]):continue
                        prev=ids_by_frame[t-1]
                        # Avoid immediate splits/births and ambiguous competing boxes.
                        if len(prev)!=len(ids):continue
                        overlap=iou_matrix(d['boxes'][prev],d['boxes'][ids]);high=overlap>=.7;possible=overlap>=.2
                        for i,j in zip(*np.where(high)):
                            if possible[i].sum()==1 and possible[:,j].sum()==1:positives.append(pair(prev[i],ids[j],'positive_candidate'))
                    chosen=sorted(set(round((len(negatives)-1)*q) for q in [0,.5,1])) if negatives else []
                    selected=[negatives[i] for i in chosen]
                    for row in selected:
                        a,b=row['detection_a'],row['detection_b'];impath=images[row['frame_a']];negative_review.append({**row,'image_path':str(impath.relative_to(Path(cfg['data_root']))),'source_image_sha256':sha(impath),'boxes':[d['boxes'][a].tolist(),d['boxes'][b].tolist()],'confidence':d['confidence'][[a,b]].tolist(),'status':'unreviewed'})
                    rows.extend(positives+selected);counts.append({'scene':scene,'sequence':seq,'partition':group,'role':role,'positive_candidates':len(positives),'all_negative_candidates':len(negatives),'negative_frames_selected_for_review':len(selected)})
    with (OUT/'pair_candidates.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    boards=ART/'pair_review';boards.mkdir(exist_ok=True)
    for offset in range(0,len(negative_review),9):
        group=negative_review[offset:offset+9];im=Image.new('RGB',(1536,1632),'#151515');draw=ImageDraw.Draw(im)
        for cell,row in enumerate(group):
            x=(cell%3)*512;y=(cell//3)*544;original=Image.open(Path(cfg['data_root'])/row['image_path']).convert('RGB').resize((512,512));im.paste(original,(x,y+32));draw.text((x+5,y+6),row['id']+' '+row['partition'].removeprefix('representation_'),fill='white')
            for j,box in enumerate(row['boxes']):
                b=[x+box[0]*512,y+32+box[1]*512,x+box[2]*512,y+32+box[3]*512];draw.rectangle(b,outline=['#2de0dc','#ffe34d'][j],width=3);draw.text((b[0]+2,b[1]+2),str(j+1),fill=['#2de0dc','#ffe34d'][j])
            row['board']=f'negative_{offset//9:02}.jpg'
        im.save(boards/f'negative_{offset//9:02}.jpg',quality=92)
    write(OUT/'negative_pair_review_drafts.json',negative_review)
    write(OUT/'pair_inventory.json',{'normal_only':True,'track_ids_used_as_labels':False,'source_feature_sha256':source_files,'csv_sha256':sha(OUT/'pair_candidates.csv'),'negative_drafts_sha256':sha(OUT/'negative_pair_review_drafts.json'),'counts':counts,'negative_frames_to_review':len(negative_review),'rules':{'confidence':.5,'positive_adjacent_iou':.7,'positive_competing_iou':.2,'positive_equal_role_box_count':True,'negative_same_frame_iou_max':.1,'max_negative_frames_per_video_role':3},'boards':{str(p):sha(p) for p in sorted(boards.glob('*.jpg'))},'interpretation':'Geometric candidates only. Simultaneously separated material pieces may be distinct objects; overlapping/split boxes or background false detections must not become identity negatives. No test data or identity ground truth used.'})
    print('Pair inventory:',len(rows),'rows;',len(negative_review),'negative frames to review')

if __name__=='__main__':main()
