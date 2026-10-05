"""Fit independent phase heads on normal weak targets, not test labels."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ipad_vad.learned_phase import PhaseHead,causal_visual_input
from ipad_vad.learned_detector import sha,write
from phase43_common import OUT,ART,SCENES,HEADS
from experiment41_phases import load_npz

def main():
    if (OUT/'training.json').exists():raise RuntimeError('Training completed')
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            s=str(args[0])
            if '/test_label/' in s or '/testing/' in s or Path(s).name.startswith('testing_'):raise RuntimeError('Normal only')
    sys.addaudithook(guard);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;cfg=json.loads(Path('configs/experiment43_phase.json').read_text());audit=json.loads((OUT/'target_audit.json').read_text());review=json.loads((OUT/'target_review.json').read_text());assert review['reviewed_frames']==60;split=json.loads(Path('results/experiment40/data_manifest.json').read_text())['subsplits'];files=[Path(__file__),Path('scripts/phase43_common.py'),Path('scripts/prepare_phase43.py'),Path('src/ipad_vad/learned_phase.py'),Path('configs/experiment43_phase.json'),Path('docs/EXPERIMENT43_PLAN.md'),Path('tests/test_learned_phase.py'),OUT/'target_audit.json',OUT/'target_review.json',OUT/'teacher_fit.json',ART/'teacher_centers.npz',ART/'R02_relation_model.npz'];files+=sorted((ART/'targets').glob('*/*.npz'));data={}
    for scene in SCENES:
        for group in ['representation_train','representation_validation']:
            xx=[];yy=[]
            for seq in split[scene][group]:
                p=Path('artifacts/experiment41/detections/features')/scene/f'training_{seq}.npz';assert sha(p)==audit['source_sha256'][str(p)];d=load_npz(p);target=ART/'targets'/scene/f'training_{seq}.npz';r=next(r for r in audit['sequences'] if r['scene']==scene and r['sequence']==seq);assert sha(target)==r['target_sha256'];t=load_npz(target);x=causal_visual_input(d,window=cfg['window']);xx.append(x[t['keep']]);yy.append(t['labels'][t['keep']])
            x=np.concatenate(xx);y=np.concatenate(yy);k=3 if scene=='R01' else 4;assert set(y.tolist())==set(range(k));data[(scene,group)]=(torch.tensor(x,device=cfg['device']),torch.tensor(y,device=cfg['device'],dtype=torch.long));assert np.isfinite(x).all()
    # Pre-optimizer snapshot; data source hashes prove backbone features unchanged.
    write(OUT/'training_protocol.json',{'normal_only':True,'optimizer_and_checkpoint_calibration_excluded':True,'file_sha256':{str(p):sha(p) for p in files},'source_feature_sha256':audit['source_sha256'],'classes':{s:3 if s=='R01' else 4 for s in SCENES}});dest=ART/'heads';dest.mkdir(exist_ok=True);rows=[];allhistory={}
    for scene in SCENES:
        k=3 if scene=='R01' else 4;x,y=data[(scene,'representation_train')];vx,vy=data[(scene,'representation_validation')];state_indices=[torch.where(y==i)[0].cpu().numpy() for i in range(k)]
        for branch in HEADS:
            torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);m=PhaseHead(k,rank=cfg['rank'],adapt=branch=='adapter').to(cfg['device']);initial={n:v.detach().clone() for n,v in m.state_dict().items()};opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);history=[];best=float('inf');bestepoch=0;start=time.perf_counter();maxgrad=0;steps=0
            def evaluate(ex,ey):
                with torch.no_grad():
                    logits,z=m(ex);ce=F.cross_entropy(logits,ey,reduction='none');pred=logits.argmax(1);conf=torch.zeros((k,k),dtype=torch.long,device=ex.device);conf.index_put_((ey,pred),torch.ones_like(ey),accumulate=True);loss=float(torch.stack([ce[ey==i].mean() for i in range(k)]).mean());agree=float((pred==ey).float().mean());bal=float(torch.stack([(pred[ey==i]==i).float().mean() for i in range(k)]).mean());pres=float((z-ex).square().sum(1).mean())
                return {'macro_cross_entropy':loss,'weak_target_agreement':agree,'macro_weak_target_recall':bal,'confusion':conf.cpu().tolist(),'preservation':pres}
            for epoch in range(cfg['epochs']+1):
                if epoch:
                    m.train()
                    for step in range(cfg['steps_per_epoch']):
                        classes=rng.integers(0,k,cfg['batch_size']);ix=np.array([rng.choice(state_indices[c]) for c in classes]);logits,z=m(x[ix]);loss=F.cross_entropy(logits,y[ix])+cfg['preservation_weight']*(z-x[ix]).square().sum(1).mean();opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['gradient_clip']));assert torch.isfinite(loss) and np.isfinite(gn);maxgrad=max(maxgrad,gn);opt.step();steps+=1
                m.eval();tr=evaluate(x,y);va=evaluate(vx,vy);history.append({'epoch':epoch,'train':tr,'validation':va})
                if va['macro_cross_entropy']<best:best=va['macro_cross_entropy'];bestepoch=epoch;torch.save(m.state_dict(),dest/f'{scene}_{branch}.pt')
            elapsed=time.perf_counter()-start;m.load_state_dict(torch.load(dest/f'{scene}_{branch}.pt',map_location=cfg['device'],weights_only=True));delta={n:float(torch.linalg.vector_norm(v-initial[n])) for n,v in m.state_dict().items()};assert maxgrad>0 and (sum(delta.values())>0 or bestepoch==0);selected=history[bestepoch];row={'scene':scene,'branch':branch,'seed':cfg['seed'],'selected_epoch':bestepoch,'parameters':sum(p.numel() for p in m.parameters()),'optimizer_steps':steps,'seconds_training_validation':elapsed,'max_gradient_norm':maxgrad,'parameter_delta_l2':delta,'checkpoint':str(dest/f'{scene}_{branch}.pt'),'checkpoint_sha256':sha(dest/f'{scene}_{branch}.pt'),'selected_train':selected['train'],'selected_validation':selected['validation'],'train_samples':len(y),'validation_samples':len(vy)};rows.append(row);allhistory[f'{scene}_{branch}']=history;print(scene,branch,'epoch',bestepoch,'CE',best,'agreement',selected['validation']['weak_target_agreement'],flush=True)
    protocol=json.loads((OUT/'training_protocol.json').read_text())
    for mapping in ['file_sha256','source_feature_sha256']:
        for p,h in protocol[mapping].items():assert sha(p)==h,p
    write(OUT/'history.json',allhistory);write(OUT/'training.json',{'normal_only':True,'rows':rows,'source_feature_sha256_unchanged':True,'all_backbones_frozen_not_loaded':True,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'training_protocol_sha256':sha(OUT/'training_protocol.json'),'history_sha256':sha(OUT/'history.json'),'interpretation':'Agreement with weak teacher targets only; no action GT accuracy. Four independent process heads per branch, one optimizer seed; downstream appearance encoders unchanged.'})
if __name__=='__main__':main()
