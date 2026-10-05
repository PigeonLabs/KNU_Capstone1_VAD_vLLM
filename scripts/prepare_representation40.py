"""Inventory normal views and split whole videos before visual optimization."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from ipad_vad.data import frames_in_order

OUT=Path('results/experiment40');ART=Path('artifacts/experiment40')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def main():
    cfg=json.loads(Path('configs/experiment40_representation.json').read_text());split=json.loads(Path('results/stage00/splits.json').read_text());rng=np.random.default_rng(cfg['split_seed']);rows=[];views=[];subsplits={};files={}
    for scene in cfg['scenes']:
        fit=split[scene]['fit'];cal=split[scene]['calibration'];nval=math.ceil(len(fit)*cfg['representation_validation_fraction']);val=sorted(rng.choice(fit,nval,replace=False).tolist());train=[s for s in fit if s not in val]
        assert set(train).isdisjoint(val) and set(train+val).isdisjoint(cal)
        subsplits[scene]={'representation_train':train,'representation_validation':val,'normal_calibration':cal,'downstream_fit':fit}
        for seq in fit+cal:
            path=Path(cfg['source_features'][scene])/f'training_{seq}.npz'
            with np.load(path,allow_pickle=False) as a:d=dict(a)
            images=frames_in_order(Path(cfg['data_root'])/scene/'training/frames'/seq)
            assert int(d['frame_count'])==len(images) and len(d['global_features'])==len(d['indices'])
            files[str(path)]=sha(path);part='representation_train' if seq in train else 'representation_validation' if seq in val else 'normal_calibration'
            counts={'global':0,'crop':0}
            for kind in ['global','crop']:
                for j in range(len(d['indices']) if kind=='global' else len(d['boxes'])):
                    step=j if kind=='global' else int(d['object_frames'][j]);frame=int(d['indices'][step]);box=None if kind=='global' else d['boxes'][j].tolist()
                    if box is not None:assert np.isfinite(box).all() and 0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1
                    views.append({'scene':scene,'sequence':seq,'partition':part,'kind':kind,'sample_index':step,'source_frame':frame,'detection_index':None if kind=='global' else j,'box':box,'path':str(images[frame])});counts[kind]+=1
            rows.append({'scene':scene,'sequence':seq,'partition':part,'source_frames':len(images),'views':counts,'source_feature_sha256':files[str(path)]})
    destination=ART/'normal_views.jsonl';destination.parent.mkdir(parents=True,exist_ok=True)
    text=''.join(json.dumps(v,separators=(',',':'))+'\n' for v in views)
    if destination.exists():assert destination.read_text()==text,'Refuse to replace an existing view manifest'
    else:destination.write_text(text)
    model=json.loads(Path(cfg['model_record']).read_text());base=Path(model['local_path'])
    model['sha256']={name:sha(base/name) for name in ['open_clip_model.safetensors','open_clip_config.json']}
    write(OUT/'data_manifest.json',{'normal_only':True,'subsplits':subsplits,'sequences':rows,'view_manifest':str(destination),'view_manifest_sha256':sha(destination),'normal_source_feature_sha256':files,'model':model,'total_normal_videos':len(rows),'view_count':len(views),'calibration_views_excluded_from_encoder':sum(r['partition']=='normal_calibration' for r in views),'original_splits_sha256':sha('results/stage00/splits.json'),'note':'Whole-video split; independent recording groups unknown. Cached detector and phase outputs are fixed and shared by A-D; no learned ReID or phase supervision.'})
    print(json.dumps({'subsplits':subsplits,'normal_videos':len(rows),'views':len(views)},indent=2),flush=True)

if __name__=='__main__':main()
