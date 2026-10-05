"""Deterministic normal-calibration visual diagnostics; no new training annotations."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from ipad_vad.data import frames_in_order
from ipad_vad.learned_detector import sha,write
from experiment41_phases import load_npz

def main():
    cfg=json.loads(Path('configs/experiment40_representation.json').read_text());manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text());dest=Path('artifacts/experiment41/normal_output_review');dest.mkdir(parents=True,exist_ok=True);rows=[]
    assert Path('results/experiment41/detector_training_extraction.json').exists()
    colors=['#ff4f7b','#27e3d1','#ffe34d']
    for scene in cfg['scenes']:
        calibration=manifest['subsplits'][scene]['normal_calibration']
        for seq in [calibration[0],calibration[-1]]:
            frames=frames_in_order(Path(cfg['data_root'])/scene/'training/frames'/seq)
            arrays={arm:load_npz(root/f'training_{seq}.npz') for arm,root in [('frozen',Path(cfg['source_features'][scene])),('learned',Path('artifacts/experiment41/detections/features')/scene)]}
            n=len(arrays['frozen']['indices']);steps=[round((n-1)*q) for q in [.2,.5,.8]];board=Image.new('RGB',(1024,1632),'#151515');draw=ImageDraw.Draw(board)
            for row,step in enumerate(steps):
                frame=int(arrays['frozen']['indices'][step]);img=Image.open(frames[frame]).convert('RGB').resize((512,512))
                for col,arm in enumerate(['frozen','learned']):
                    data=arrays[arm];assert int(data['indices'][step])==frame;x=col*512;y=row*544;board.paste(img,(x,y+32));draw.text((x+5,y+7),f'{scene}/{seq} frame {frame} | {arm} | phase {data["phases"][step]}',fill='white')
                    boxes=[]
                    for i in np.flatnonzero(data['object_frames']==step):
                        role=int(data['roles'][i]);b=data['boxes'][i];coords=[x+b[0]*512,y+32+b[1]*512,x+b[2]*512,y+32+b[3]*512];draw.rectangle(coords,outline=colors[role],width=2);draw.text((coords[0]+2,coords[1]+2),f'r{role} id{data["tracks"][i]} {data["confidence"][i]:.2f}',fill=colors[role]);boxes.append({'role':role,'track':int(data['tracks'][i]),'box':b.tolist(),'confidence':float(data['confidence'][i])})
                    rows.append({'scene':scene,'sequence':seq,'frame':frame,'sample_index':step,'arm':arm,'phase':int(data['phases'][step]),'boxes':boxes,'source_image_sha256':sha(frames[frame])})
            board.save(dest/f'{scene}_{seq}.jpg',quality=92)
    write('results/experiment41/normal_output_review_inventory.json',{'selection':'first and last normal-calibration sequence per process, sample fractions .2/.5/.8, chosen without test labels','purpose':'post-training qualitative diagnostics only, no new optimization or checkpoint selection','frames':24,'rows':rows,'boards':{str(p):sha(p) for p in sorted(dest.glob('*.jpg'))}})

if __name__=='__main__':main()
