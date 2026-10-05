"""Normal-only contact sheets for model-assisted pseudo-label review (not GT)."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ipad_vad.data import frames_in_order

OUT=Path('artifacts/experiment41/review');OUT.mkdir(parents=True,exist_ok=True)
cfg=json.load(open('configs/experiment40_representation.json'));manifest=json.load(open('results/experiment40/data_manifest.json'));records=[]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
for scene in cfg['scenes']:
 canvas=Image.new('RGB',(1920,840),'white');items=[]
 for part in ['representation_train','representation_validation']:
  seqs=manifest['subsplits'][scene][part]
  for seq in [seqs[0],seqs[len(seqs)//2],seqs[-1]]:
   with np.load(Path(cfg['source_features'][scene])/f'training_{seq}.npz') as f:d=dict(f)
   step=len(d['indices'])//2;frame=int(d['indices'][step]);path=frames_in_order(Path(cfg['data_root'])/scene/'training/frames'/seq)[frame]
   image=Image.open(path).convert('RGB');w,h=image.size;image=image.resize((640,360));draw=ImageDraw.Draw(image)
   ids=np.flatnonzero(d['object_frames']==step)
   for j in ids:
    role=int(d['roles'][j]);box=d['boxes'][j]*np.array([640,360,640,360]);color=['red','lime','cyan'][role]
    draw.rectangle(tuple(box),outline=color,width=2);draw.text((box[0],max(0,box[1]-19)),f'{j}:r{role} {d["confidence"][j]:.2f}',font=font,fill=color,stroke_width=1,stroke_fill='black')
   row={'scene':scene,'partition':part,'sequence':seq,'step':step,'frame':frame,'path':str(path),'size':[w,h],'detections':[{'index':int(j),'role':int(d['roles'][j]),'confidence':float(d['confidence'][j]),'box':d['boxes'][j].tolist()} for j in ids]};items.append(row)
 for i,(row) in enumerate(items):
  image=Image.open(row['path']).convert('RGB').resize((640,360));draw=ImageDraw.Draw(image)
  for d in row['detections']:
   box=np.array(d['box'])*[640,360,640,360];color=['red','lime','cyan'][d['role']];draw.rectangle(tuple(box),outline=color,width=2);draw.text((box[0],max(0,box[1]-19)),f'{d["index"]}:r{d["role"]} {d["confidence"]:.2f}',font=font,fill=color,stroke_width=1,stroke_fill='black')
  x=(i%3)*640;y=(i//3)*420;canvas.paste(image,(x,y));draw=ImageDraw.Draw(canvas);draw.text((x+5,y+365),f'{scene}/{row["sequence"]} frame{row["frame"]} {row["partition"]}',fill='black',font=font)
 canvas.save(OUT/f'{scene}.jpg');records.extend(items)
(OUT/'frames.json').write_text(json.dumps(records,indent=2)+'\n')
