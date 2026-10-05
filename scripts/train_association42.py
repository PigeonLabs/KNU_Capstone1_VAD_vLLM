"""Normal-only metric adapter training; freeze protocol before any optimizer step."""
import csv,json,sys,time,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from ipad_vad.learned_detector import sha,write
from ipad_vad.learned_association import ResidualMetric,pair_terms
OUT=Path('results/experiment42');ART=Path('artifacts/experiment42')

def load_pairs(partition):
    rows=[r for r in csv.DictReader((OUT/'accepted_pairs.csv').open()) if r['partition']==partition];cache={};a=[];b=[];groups=[];neg=[]
    inventory=json.loads((OUT/'pair_inventory.json').read_text())
    for r in rows:
        key=(r['scene'],r['sequence'])
        if key not in cache:
            p=Path('artifacts/experiment41/detections/features')/key[0]/f'training_{key[1]}.npz';assert sha(p)==inventory['source_feature_sha256'][str(p)]
            with np.load(p) as f:cache[key]=f['crop_features']
        a.append(cache[key][int(r['detection_a'])]);b.append(cache[key][int(r['detection_b'])]);groups.append(r['scene']+'/'+r['role']);neg.append(r['kind']=='negative_candidate')
    return np.array(a,dtype=np.float32),np.array(b,dtype=np.float32),np.array(neg),groups,rows

def main():
    if (OUT/'training.json').exists():raise RuntimeError('Preserve completed training')
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            s=str(args[0])
            if '/test_label/' in s or '/testing/' in s or Path(s).name.startswith('testing_'):raise RuntimeError('Normal only')
    sys.addaudithook(guard);cfg=json.loads(Path('configs/experiment42_association.json').read_text());torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(cfg['device']);datasets={}
    for part in ['representation_train','representation_validation']:
        a,b,n,g,rows=load_pairs(part);datasets[part]=(torch.tensor(a,device=device),torch.tensor(b,device=device),n,g)
    train=datasets['representation_train'];val=datasets['representation_validation'];model=ResidualMetric(rank=cfg['rank']).to(device);initial={k:v.detach().clone() for k,v in model.state_dict().items()};opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);rng=np.random.default_rng(cfg['seed']);positive_groups={g:np.array([i for i,x in enumerate(train[3]) if x==g and not train[2][i]]) for g in sorted(set(train[3]))};positive_groups={k:v for k,v in positive_groups.items() if len(v)};negative=np.flatnonzero(train[2]);groupnames=list(positive_groups)
    def objective(data):
        with torch.no_grad():
            p,n,preserve=pair_terms(model,data[0],data[1],cfg['negative_margin']);pos=[float(p[np.array([i for i,g in enumerate(data[3]) if g==key and not data[2][i]])].mean()) for key in sorted(set(data[3])) if any(g==key and not data[2][i] for i,g in enumerate(data[3]))];neg=float(n[data[2]].mean());pres=(float(preserve[~data[2]].mean())+float(preserve[data[2]].mean()))/2
        return {'loss':float(np.mean(pos))+neg+cfg['preservation_weight']*pres,'positive_group_macro':float(np.mean(pos)),'negative':neg,'preservation':pres}
    # Smoke: initial identity, finite nonzero gradient without taking a step.
    np.testing.assert_allclose(model(train[0][:8]).detach().cpu(),torch.nn.functional.normalize(train[0][:8],dim=-1).cpu(),atol=1e-6);p,n,pr=pair_terms(model,train[0][negative],train[1][negative]);smoke=n.mean()+pr.mean();smoke.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad']));assert grad>0 and np.isfinite(grad);opt.zero_grad(set_to_none=True)
    files=[Path(__file__),Path('src/ipad_vad/learned_association.py'),Path('src/ipad_vad/tracking.py'),Path('configs/experiment42_association.json'),Path('docs/EXPERIMENT42_PLAN.md'),Path('results/experiment40/data_manifest.json')]+[OUT/n for n in ['accepted_pairs.csv','pair_selection.json','pair_inventory.json','negative_pair_review.json','positive_pair_review.json']]
    for p,h in json.loads((OUT/'pair_selection.json').read_text())['review_sha256'].items():assert sha(p)==h
    protocol={'normal_only':True,'calibration_used_for_gradient_or_checkpoint':False,'file_sha256':{str(p):sha(p) for p in files},'smoke_gradient_norm':grad,'trainable_parameters':sum(p.numel() for p in model.parameters()),'feature_sources':json.loads((OUT/'pair_inventory.json').read_text())['source_feature_sha256']};write(OUT/'training_protocol.json',protocol)
    dest=ART/'adapter';dest.mkdir(parents=True,exist_ok=True);history=[];best=float('inf');epochbest=0;start=time.perf_counter();steps=0;maxgrad=0
    for epoch in range(cfg['epochs']+1):
        if epoch:
            model.train()
            for step in range(cfg['steps_per_epoch']):
                selectedgroups=rng.choice(groupnames,cfg['batch_size']//2);pi=np.array([rng.choice(positive_groups[g]) for g in selectedgroups]);ni=rng.choice(negative,cfg['batch_size']//2,replace=True);ix=np.concatenate([pi,ni]);p,n,pr=pair_terms(model,train[0][ix],train[1][ix],cfg['negative_margin']);half=len(pi);loss=p[:half].mean()+n[half:].mean()+cfg['preservation_weight']*pr.mean();opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad']));assert np.isfinite(gn) and torch.isfinite(loss);maxgrad=max(maxgrad,gn);opt.step();steps+=1
        model.eval();tr=objective(train);va=objective(val);history.append({'epoch':epoch,'train':tr,'validation':va})
        if va['loss']<best:best=va['loss'];epochbest=epoch;torch.save(model.state_dict(),dest/'best.pt')
        print(epoch,'train',round(tr['loss'],6),'val',round(va['loss'],6),'best',epochbest,flush=True)
    seconds=time.perf_counter()-start;model.load_state_dict(torch.load(dest/'best.pt',map_location=device,weights_only=True));delta={k:float(torch.linalg.vector_norm(v-initial[k])) for k,v in model.state_dict().items()};assert sum(delta.values())>0 or epochbest==0
    for p,h in protocol['file_sha256'].items():assert sha(p)==h
    for p,h in protocol['feature_sources'].items():assert sha(p)==h
    write(OUT/'history.json',history);write(OUT/'training.json',{'seed':cfg['seed'],'selected_epoch':epochbest,'selected_val':best,'epochs_completed':cfg['epochs'],'optimizer_steps':steps,'seconds_training_and_validation':seconds,'parameters':protocol['trainable_parameters'],'max_gradient_norm':maxgrad,'parameter_delta_l2':delta,'checkpoint':str(dest/'best.pt'),'checkpoint_sha256':sha(dest/'best.pt'),'normal_feature_sha256_unchanged':True,'frozen_backbones_not_loaded_or_updated':True,'history_sha256':sha(OUT/'history.json'),'training_protocol_sha256':sha(OUT/'training_protocol.json'),'device':str(device),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None})
if __name__=='__main__':main()
