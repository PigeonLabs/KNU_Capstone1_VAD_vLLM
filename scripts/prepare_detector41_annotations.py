"""Draft weak normal boxes for exhaustive model-assisted visual review, never GT."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ipad_vad.data import frames_in_order
from ipad_vad.tracking import iou_matrix

OUT=Path('artifacts/experiment41/annotation_review');OUT.mkdir(parents=True,exist_ok=True)
CFG=json.load(open('configs/experiment40_representation.json'));MAN=json.load(open('results/experiment40/data_manifest.json'));FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)


def proposal(scene,image,d,step):
 ids=np.flatnonzero(d['object_frames']==step);boxes=[]
 def add(role,b,source):
  b=np.clip(np.asarray(b,float),0,1)
  if b[2]>b[0] and b[3]>b[1]:boxes.append({'role':role,'box':b.tolist(),'proposal_source':source})
 def candidates(role):return sorted([int(j) for j in ids if d['roles'][j]==role],key=lambda j:-d['confidence'][j])
 if scene=='R01':
  a=np.array(image.resize((256,256)));r,g,b=[a[:,:,i].astype(float) for i in range(3)];mask=((r<85)&(g<95)&(b<95)&(g-r<25)).astype('uint8');mask[:65]=0;mask[159:]=0
  n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8);valid=[v for v in stats[1:] if v[4]>=12 and v[2]<70 and v[3]<80]
  if valid:
   x,y,w,h,area=max(valid,key=lambda v:v[4]);add(0,[(x-3)/256,(y-18)/256,(x+w+4)/256,(y+h+3)/256],'normal_belt_ROI_dark_component_draft')
  add(1,[0,.25,1,.625],'normal_fixed_scene_visible_belt_draft');add(2,[0,.625,1,.785],'normal_fixed_scene_front_rail_draft')
 elif scene=='R02':
  a=[j for j in candidates(0) if .35<d['boxes'][j,[0,2]].mean()<.65 and np.prod(d['boxes'][j,2:]-d['boxes'][j,:2])>.08]
  if a:
   # Prefer the top plate over the lower metal base that shares words in the prompt.
   j=min(a,key=lambda j:d['boxes'][j,1]);add(0,d['boxes'][j],'teacher_top_plate_candidate')
  a=[j for j in candidates(1) if .38<d['boxes'][j,[0,2]].mean()<.64 and d['boxes'][j,0]>.27 and d['boxes'][j,2]<.8]
  if a:add(1,d['boxes'][a[0]],'teacher_center_scissor_candidate_below_05_review_required')
  a=[j for j in candidates(2) if .34<d['boxes'][j,[0,2]].mean()<.65]
  if a:add(2,d['boxes'][a[0]],'teacher_instrument_candidate')
 elif scene=='R03':
  for role in [0,1]:
   a=candidates(role)
   if a:add(role,d['boxes'][a[0]],'teacher_top_role_candidate')
 else:
  a=candidates(0);chosen=[]
  for j in a:
   b=d['boxes'][j]
   if b[0]<.75 and np.prod(b[2:]-b[:2])>=.025 and (not chosen or np.max(iou_matrix([b],chosen))<.25):add(0,b,'teacher_sheet_candidate');chosen.append(b)
  add(1,[.53,0,.79,.36],'manual_seed_upright_blade_draft_review_required')
  add(2,[0,.26,.80,.92],'normal_fixed_scene_trough_draft')
 return boxes


def main():
 records=[]
 for scene in CFG['scenes']:
  for part in ['representation_train','representation_validation']:
   for seq in MAN['subsplits'][scene][part]:
    with np.load(Path(CFG['source_features'][scene])/f'training_{seq}.npz') as f:d=dict(f)
    images=frames_in_order(Path(CFG['data_root'])/scene/'training/frames'/seq)
    for fraction in [.2,.5,.8]:
     step=min(len(d['indices'])-1,int(len(d['indices'])*fraction));frame=int(d['indices'][step]);path=images[frame];image=Image.open(path).convert('RGB')
     records.append({'id':f'{scene}_{seq}_{frame:04}','scene':scene,'sequence':seq,'partition':part,'frame':frame,'sample_index':step,'path':str(path),'width':image.width,'height':image.height,'annotations':proposal(scene,image,d,step),'status':'unreviewed','reviewer':'pending','source':'normal_only_model_assisted_weak_annotation_not_ground_truth'})
 dest=OUT/'drafts.json';dest.write_text(json.dumps(records,indent=2)+'\n')
 for scene in CFG['scenes']:
  selected=[r for r in records if r['scene']==scene]
  for page,start in enumerate(range(0,len(selected),9)):
   batch=selected[start:start+9];canvas=Image.new('RGB',(1536,1728),'white');draw=ImageDraw.Draw(canvas)
   for i,row in enumerate(batch):
    image=Image.open(row['path']).convert('RGB').resize((512,512));d=ImageDraw.Draw(image)
    for j,a in enumerate(row['annotations']):
     box=np.array(a['box'])*512;color=['red','lime','cyan'][a['role']];d.rectangle(tuple(box),outline=color,width=2);d.text((box[0],max(0,box[1]-18)),f'{j}/r{a["role"]}',font=FONT,fill=color,stroke_width=1,stroke_fill='black')
    x=(i%3)*512;y=(i//3)*576;canvas.paste(image,(x,y));draw.text((x+4,y+514),f'{row["id"]} {row["partition"]}',fill='black',font=FONT)
   canvas.save(OUT/f'{scene}_{page:02}.jpg')
 print(len(records),{s:sum(r['scene']==s for r in records) for s in CFG['scenes']})

if __name__=='__main__':main()
