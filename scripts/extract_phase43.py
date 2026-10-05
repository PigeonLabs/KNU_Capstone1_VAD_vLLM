"""Replay process-specific phase heads without changing shared visual features."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from ipad_vad.learned_detector import sha,write
from ipad_vad.dwell_episodes import extract_dwell_episodes
from phase43_common import OUT,ART,SCENES,BRANCHES,teachers,head,derived
from experiment41_phases import load_npz
from experiment40_normal import RUNS

def main(split):
    if (OUT/f'{split}_observations.json').exists():raise RuntimeError('Phase replay already frozen')
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            s=str(args[0])
            if '/test_label/' in s or (split=='training' and ('/testing/' in s or Path(s).name.startswith('testing_'))):raise RuntimeError('Forbidden label/test access')
    sys.addaudithook(guard);torch.set_num_threads(4);train=json.loads((OUT/'training.json').read_text());protocol=json.loads((OUT/'training_protocol.json').read_text())
    for p,h in protocol['file_sha256'].items():assert sha(p)==h,p
    for r in train['rows']:assert sha(r['checkpoint'])==r['checkpoint_sha256']
    if split=='testing':
        for p,h in json.loads((OUT/'normal_models_checkpoint.json').read_text())['file_sha256'].items():assert sha(p)==h,p
    ms,centers=teachers();models={s:{b:head(s,b) for b in ['linear','adapter']} for s in SCENES};source=json.loads(Path(f'results/experiment41/detector_{split}_extraction.json').read_text());files={};rows=[];start=time.perf_counter();manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text())['subsplits'];aud=json.loads((OUT/'target_audit.json').read_text());checks=0
    for r in source['sequences']:
        scene,seq=r['scene'],r['sequence'];p=Path('artifacts/experiment41/detections/features')/scene/f'{split}_{seq}.npz';assert sha(p)==r['sha256'];files[str(p)]=r['sha256'];d=load_npz(p);new,valid,chosen=derived(scene,d,ms,centers,models[scene]);part=next(g for g,ss in manifest[scene].items() if g in ['representation_train','representation_validation','normal_calibration'] and seq in ss) if split=='training' else 'test'
        if split=='training' and part!='normal_calibration':
            t=load_npz(ART/'targets'/scene/p.name);np.testing.assert_array_equal(new['teacher']['phases'],t['teacher_phases']);np.testing.assert_array_equal(valid,t['valid']);np.testing.assert_array_equal(chosen,t['chosen']);checks+=1
        for run in RUNS:
            q=Path('artifacts/experiment41')/run/'features'/scene/p.name;record=json.loads(Path(f'results/experiment41/{run}_{split}_extraction.json').read_text());expected=next(x['sha256'] for x in record['sequences'] if x['scene']==scene and x['sequence']==seq);assert sha(q)==expected;files[str(q)]=expected
        for b,value in new.items():
            meta={k:v for k,v in value.items() if k not in ['global_features','crop_features']}
            for k in meta:
                if k!='phases':np.testing.assert_array_equal(meta[k],d[k])
            dest=ART/b/'observations'/scene/p.name;dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**meta);files[str(dest)]=sha(dest);k=3 if scene=='R01' else 4;episodes=extract_dwell_episodes(value)['episodes'] if scene in ['R03','R04'] else []
            rows.append({'scene':scene,'sequence':seq,'partition':part,'branch':b,'samples':len(valid),'emission_valid_samples':int(valid.sum()),'phase_changed_vs_41':int((value['phases']!=d['phases']).sum()),'phase_changed_vs_teacher':int((value['phases']!=new['teacher']['phases']).sum()),'phase_counts':np.bincount(value['phases'],minlength=k).tolist(),'transitions':int((np.diff(value['phases'])!=0).sum()),'episodes':episodes,'metadata_sha256':sha(dest)})
        print(split,scene,seq,'phase changes',*[rr['phase_changed_vs_41'] for rr in rows[-3:]],flush=True)
    files.update({str(p):sha(p) for p in [Path(__file__),Path('scripts/phase43_common.py'),Path('src/ipad_vad/learned_phase.py'),OUT/'training.json',OUT/'training_protocol.json',ART/'teacher_centers.npz',ART/'R02_relation_model.npz']});files.update({r['checkpoint']:r['checkpoint_sha256'] for r in train['rows']});write(OUT/f'{split}_observations.json',{'labels_opened':False,'normal_only':split=='training','file_sha256':files,'sequences':rows,'training_target_reconstruction_checks':checks,'seconds_replay_including_io_hashing':time.perf_counter()-start,'shared_features_boxes_tracks_other_metadata_unchanged':True})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['training','testing'],required=True);main(p.parse_args().split)
