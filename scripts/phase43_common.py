"""Frozen teacher observations, head restoration and visual-only input contract."""
import json
from pathlib import Path
import numpy as np
import torch
from ipad_vad.relational_phase import RelationalPhase
from ipad_vad.learned_phase import PhaseHead,causal_visual_input,causal_hold
from experiment41_phases import phase_models,load_npz
OUT=Path('results/experiment43');ART=Path('artifacts/experiment43')
SCENES=['R01','R02','R03','R04'];HEADS=['linear','adapter'];BRANCHES=['teacher',*HEADS]

def teachers():
    ms=phase_models();p=load_npz(ART/'R02_relation_model.npz');m=RelationalPhase(anchor_role=0,target_role=1,k=4,smoothing=3,seed=42)
    for k in ['location','scale','centers']:setattr(m,k,p[k])
    m.area_upper=dict(zip(p['area_roles'].tolist(),p['area_upper'].tolist()));ms['R02']=m;cs=load_npz(ART/'teacher_centers.npz');return ms,cs

def observations(scene,d,ms,centers):
    if scene=='R01':
        _,valid,chosen,positions=ms[scene].transform(d);history=[];x=[]
        for pos in positions:
            if np.isfinite(pos):history.append(pos);history=history[-3:]
            x.append([np.mean(history) if history else 0.])
        x=np.array(x)
    else:
        _,valid,chosen,desc=ms[scene].transform(d);x=(desc-ms[scene].location)/ms[scene].scale
        if scene=='R04':x=np.arcsinh(x)
    distance=((x[:,None,:]-centers[scene][None,:,:])**2).sum(-1);winner=distance.argmin(1);return causal_hold(winner,valid),valid,chosen

def head(scene,branch):
    cfg=json.loads(Path('configs/experiment43_phase.json').read_text());m=PhaseHead(3 if scene=='R01' else 4,rank=cfg['rank'],adapt=branch=='adapter');m.load_state_dict(torch.load(ART/'heads'/f'{scene}_{branch}.pt',map_location='cpu',weights_only=True));m.eval();return m

def derived(scene,d,ms,centers,models):
    teacher,valid,chosen=observations(scene,d,ms,centers);out={};features=causal_visual_input(d)
    for branch in BRANCHES:
        if branch=='teacher':phase=teacher
        else:
            with torch.no_grad():pred=models[branch](torch.from_numpy(features))[0].argmax(1).numpy()
            phase=causal_hold(pred,valid)
        # R03/R04 relation descriptors and selected boxes stay identical to 41.
        # R02's new relation teacher is only a target/emission mask; baseline has no dwell.
        out[branch]={**d,'phases':phase}
    return out,valid,chosen
