"""Restore and apply the exact pre-41 phase models; never refit phase parameters."""
import json
from pathlib import Path
import numpy as np
from ipad_vad.spatial_phase import SpatialPhase
from ipad_vad.relational_phase import RelationalPhase
from experiment37_asinh_phase import models as r04_models


def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def phase_models():
    cfg=json.loads(Path('configs/experiment02.json').read_text());e=json.loads(Path('results/experiment02/phase_grounding.json').read_text())['grounding_model']
    r01=SpatialPhase(**cfg['spatial_phase']);r01.axis=0 if e['dominant_axis']=='x' else 1;r01.band=np.array(e['perpendicular_band']);r01.centers=np.array(e['phase_centers'])
    cfg=json.loads(Path('configs/experiment08.json').read_text());r03=RelationalPhase(**cfg['relational_phase']);saved=load_npz('artifacts/experiment08/relation_model.npz')
    for key in ['location','scale','centers']:setattr(r03,key,saved[key])
    r03.area_upper=dict(zip(saved['area_roles'].tolist(),saved['area_upper'].tolist()))
    return {'R01':r01,'R03':r03,'R04':r04_models()['asinh']}


def transform(scene,data,models):
    d=dict(data)
    if scene=='R01':d['phases']=models[scene].transform(d)[0]
    elif scene in ['R03','R04']:
        model=models[scene]
        if scene=='R04':d['anchor_margin']=model.raw_margins(d);d['anchor_gate_margin']=model.margins(d)
        phase,valid,chosen,x=model.transform(d);d.update(phases=phase,relation_valid=valid,relation_detection_indices=chosen,relation_descriptors=x)
    return d


def verify_control():
    cfg=json.loads(Path('configs/experiment40_representation.json').read_text());manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text());ms=phase_models();count=0
    for scene in cfg['scenes']:
        for seq in manifest['subsplits'][scene]['downstream_fit']+manifest['subsplits'][scene]['normal_calibration']:
            old=load_npz(Path(cfg['source_features'][scene])/f'training_{seq}.npz');new=transform(scene,old,ms)
            for key in ['phases','relation_valid','relation_detection_indices','relation_descriptors','anchor_margin','anchor_gate_margin']:
                if key in old:np.testing.assert_array_equal(old[key],new[key],err_msg=f'{scene}/{seq}/{key}')
            count+=1
    return count


if __name__=='__main__':print('Original normal phase caches reproduced:',verify_control())
