"""Replay frozen detections/features; store only changed tracking/phase metadata."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from ipad_vad.tracking import Tracker
from ipad_vad.learned_association import AssociationTracker,ResidualMetric
from ipad_vad.learned_detector import sha,write
from experiment41_phases import phase_models,transform,load_npz
from experiment40_normal import SCENES,RUNS
OUT=Path('results/experiment42');ART=Path('artifacts/experiment42')

def main(split):
    if (OUT/f'{split}_observations.json').exists():raise RuntimeError('Observation stage already frozen')
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            s=str(args[0])
            if '/test_label/' in s or (split=='training' and ('/testing/' in s or Path(s).name.startswith('testing_'))):raise RuntimeError('Forbidden label/test access')
    sys.addaudithook(guard);torch.set_num_threads(4);cfg=json.loads(Path('configs/experiment42_association.json').read_text());train=json.loads((OUT/'training.json').read_text());assert sha(train['checkpoint'])==train['checkpoint_sha256'];gates=json.loads((OUT/'association_gates.json').read_text());model=ResidualMetric(rank=cfg['rank']);model.load_state_dict(torch.load(train['checkpoint'],weights_only=True,map_location='cpu'));model.eval();ms=phase_models();source_record=json.loads(Path(f'results/experiment41/detector_{split}_extraction.json').read_text());rows=[];files={};controls=0;start=time.perf_counter()
    if split=='testing':
        record=json.loads((OUT/'normal_models_checkpoint.json').read_text())
        for p,h in record['file_sha256'].items():assert sha(p)==h,p
    for item in source_record['sequences']:
        scene,seq=item['scene'],item['sequence'];path=Path('artifacts/experiment41/detections/features')/scene/f'{split}_{seq}.npz';assert sha(path)==item['sha256'];files[str(path)]=item['sha256'];d=load_npz(path);original=Tracker();control=np.empty_like(d['tracks']);sample_ids=[np.flatnonzero(d['object_frames']==t) for t in range(len(d['indices']))]
        for t,ix in enumerate(sample_ids):control[ix]=original.update(d['boxes'][ix],d['roles'][ix],t)
        np.testing.assert_array_equal(control,d['tracks']);rephase=transform(scene,d,ms)
        for k in d:np.testing.assert_array_equal(d[k],rephase[k])
        controls+=1
        for run in RUNS:
            p=Path('artifacts/experiment41')/run/'features'/scene/path.name;record=json.loads(Path(f'results/experiment41/{run}_{split}_extraction.json').read_text());expected=next(r['sha256'] for r in record['sequences'] if r['scene']==scene and r['sequence']==seq);assert sha(p)==expected;files[str(p)]=expected
        with torch.no_grad():
            raw=torch.nn.functional.normalize(torch.from_numpy(d['crop_features'].astype(np.float32)),dim=-1);learned=model(raw).numpy();raw=raw.numpy()
        for branch,features in [('raw',raw),('learned',learned)]:
            role_gates={int(k.split('/')[1]):v for k,v in gates['branches'][branch]['gates'].items() if k.startswith(scene+'/')};tracker=AssociationTracker(role_gates,**cfg['association']);tracks=np.empty_like(d['tracks'])
            for t,ix in enumerate(sample_ids):tracks[ix]=tracker.update(d['boxes'][ix],d['roles'][ix],t,features[ix],d['confidence'][ix],ix)
            new=transform(scene,{**d,'tracks':tracks},ms);metadata={k:v for k,v in new.items() if k not in ['crop_features','global_features']};dest=ART/branch/'observations'/scene/path.name;dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**metadata);files[str(dest)]=sha(dest)
            changed={k:int(np.sum(np.any(new[k]!=d[k],axis=tuple(range(1,new[k].ndim))))) if new[k].ndim>1 else int(np.sum(new[k]!=d[k])) for k in new if not np.array_equal(new[k],d[k])};roles=[]
            for role in sorted(set(d['roles'])):
                ix=d['roles']==role;roles.append({'role':int(role),'old_tracks':len(set(d['tracks'][ix].tolist())),'new_tracks':len(set(tracks[ix].tolist()))})
            rows.append({'scene':scene,'sequence':seq,'branch':branch,'samples':len(d['indices']),'boxes':len(tracks),'metadata_sha256':sha(dest),'changed_elements_or_rows':changed,'phase_changed_samples':int((new['phases']!=d['phases']).sum()),'roles':roles,'appearance_edges':tracker.edges})
        print(split,scene,seq,'appearance edges',*[len(r['appearance_edges']) for r in rows[-2:]],flush=True)
    files.update({str(p):sha(p) for p in [Path(__file__),Path('src/ipad_vad/learned_association.py'),Path('scripts/experiment41_phases.py'),OUT/'association_gates.json',Path(train['checkpoint'])]});write(OUT/f'{split}_observations.json',{'labels_opened':False,'normal_only':split=='training','control_sequences_exact_to_41':controls,'seconds_replay_including_io_hashing':time.perf_counter()-start,'file_sha256':files,'sequences':rows,'feature_policy':'41 source visual/crop features remain byte-identical; only tracker-dependent phase metadata overlaid. Auxiliary association embeddings never replace anomaly or phase-anchor embeddings.'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['training','testing'],required=True);main(p.parse_args().split)
