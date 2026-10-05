"""Normal-only partial GroundingDINO fine-tuning. Separate smoke and frozen run."""
import argparse
import json
import time
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from ipad_vad.learned_detector import (load_detector, configure_detector, state_hash, adapted_state,
    restore_adapted, scene_batches, batch_inputs, sha, write, forward_loss)

OUT=Path('results/experiment41');ART=Path('artifacts/experiment41/detector')


def evaluate(model,processor,cfg,rows):
    values=defaultdict(list); details=defaultdict(lambda:defaultdict(float))
    for batch in scene_batches(rows):
        inputs,labels=batch_inputs(batch,processor,cfg)
        with torch.no_grad():out=forward_loss(model,inputs,labels)
        value=float(out.loss)
        if not np.isfinite(value):raise RuntimeError('Nonfinite validation loss')
        values[batch[0]['scene']].extend([value]*len(batch))
        for k,v in out.loss_dict.items():details[batch[0]['scene']][k]+=float(v)*len(batch)
    means={s:float(np.mean(v)) for s,v in values.items()}
    return {'macro_scene_loss':float(np.mean(list(means.values()))),'scene_loss':means,
            'scene_loss_components':{s:{k:v/len(values[s]) for k,v in d.items()} for s,d in details.items()}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=args[0].decode() if isinstance(args[0],bytes) else args[0]
            if '/test_label/' in name or '/testing/' in name or '/predictions/' in name or Path(name).name.startswith('testing_'):
                raise RuntimeError('Test access forbidden during detector training: '+name)
    sys.addaudithook(guard)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    cfg=json.loads(Path('configs/experiment41_detector.json').read_text())
    ann=json.loads(Path(cfg['annotations']).read_text());rows=[r for r in ann['records'] if r['status']=='accepted_weak_annotation']
    train=[r for r in rows if r['partition']=='representation_train'];val=[r for r in rows if r['partition']=='representation_validation']
    assert not ({(r['scene'],r['sequence']) for r in train}&{(r['scene'],r['sequence']) for r in val})
    for r in rows:assert sha(Path(cfg['data_root'])/r['path'])==r['image_sha256']
    if not args.smoke:
        protocol=json.loads((OUT/'detector_protocol.json').read_text())
        for file,h in protocol['file_sha256'].items():assert sha(file)==h,file
        if (OUT/'training.json').exists():raise RuntimeError('Completed run exists; do not overwrite')
    model,processor,record=load_detector();scope=configure_detector(model);model.cuda()
    frozen_before=state_hash(model,True);initial=adapted_state(model);initial_hash=state_hash(model,False)
    params=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    if args.smoke:
        checks=[]
        for scene in cfg['prompts']:
            batch=[r for r in train if r['scene']==scene][:2];inputs,labels=batch_inputs(batch,processor,cfg)
            optimizer.zero_grad(set_to_none=True);out=forward_loss(model,inputs,labels)
            assert torch.isfinite(out.loss);out.loss.backward()
            norms={n:float(v.grad.norm()) for n,v in model.named_parameters() if v.requires_grad and v.grad is not None}
            assert all(np.isfinite(list(norms.values()))) and any(v>0 for v in norms.values())
            assert all(v.grad is None for v in model.parameters() if not v.requires_grad)
            checks.append({'scene':scene,'ids':[r['id'] for r in batch],'loss':float(out.loss.detach()),'gradient_parameter_count':len(norms),'nonzero_gradient_parameter_count':sum(v>0 for v in norms.values()),'bbox_gradients_nonzero':any(v>0 for n,v in norms.items() if 'bbox_embed' in n),'decoder_gradients_nonzero':any(v>0 for n,v in norms.items() if n.startswith('model.decoder.'))})
            torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip']);optimizer.step()
        assert state_hash(model,True)==frozen_before
        assert state_hash(model,False)!=initial_hash
        ART.mkdir(parents=True,exist_ok=True);torch.save({'state':adapted_state(model)},ART/'smoke.pt');trained_hash=state_hash(model,False)
        restore_adapted(model,initial);assert state_hash(model,False)==initial_hash
        restore_adapted(model,torch.load(ART/'smoke.pt',weights_only=True)['state']);assert state_hash(model,False)==trained_hash
        write(OUT/'smoke.json',{'completed':True,'checks':checks,'scope':scope,'frozen_hash_unchanged':True,'adapted_hash_changed':True,'checkpoint_restore_exact':True,'model_record':record,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'training_runs_restart_from_original':True})
        print(json.dumps({'smoke':'passed','checks':checks,'trainable':scope['trainable_parameters'],'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30}),flush=True);return
    ART.mkdir(parents=True,exist_ok=True);history=[];start=time.perf_counter();best=float('inf');best_epoch=None
    for epoch in range(cfg['epochs']+1):
        train_losses=[];grad_norms=[]
        if epoch:
            rng=np.random.default_rng(cfg['seed']+epoch)
            for step,batch in enumerate(scene_batches(train,rng=rng)):
                inputs,labels=batch_inputs(batch,processor,cfg,rng=rng)
                optimizer.zero_grad(set_to_none=True);out=forward_loss(model,inputs,labels)
                if not torch.isfinite(out.loss):raise RuntimeError('Nonfinite train loss')
                out.loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip'],error_if_nonfinite=True);optimizer.step()
                train_losses.append(float(out.loss.detach()));grad_norms.append(float(norm))
                if step%20==0:print(json.dumps({'epoch':epoch,'step':step,'loss':train_losses[-1]}),flush=True)
        valid=evaluate(model,processor,cfg,val)
        row={'epoch':epoch,'optimizer_steps':len(train_losses),'train_loss_mean_batches':float(np.mean(train_losses)) if train_losses else None,'gradient_norm_mean':float(np.mean(grad_norms)) if grad_norms else None,**valid,'elapsed_seconds':time.perf_counter()-start}
        history.append(row)
        if valid['macro_scene_loss']<best:
            best=valid['macro_scene_loss'];best_epoch=epoch
            torch.save({'state':adapted_state(model),'epoch':epoch,'validation_loss':best,'scope':scope,'protocol_sha256':sha(OUT/'detector_protocol.json')},ART/'best.pt')
        torch.save({'state':adapted_state(model),'epoch':epoch,'optimizer':optimizer.state_dict()},ART/'last.pt')
        write(OUT/'training_progress.json',{'history':history,'best_epoch':best_epoch});print(json.dumps(row),flush=True)
    assert state_hash(model,True)==frozen_before
    checkpoint=torch.load(ART/'best.pt',weights_only=True);restore_adapted(model,checkpoint['state']);verify=evaluate(model,processor,cfg,val)
    assert abs(verify['macro_scene_loss']-best)<1e-6
    write(OUT/'training.json',{'completed':True,'history':history,'selected_epoch':best_epoch,'selected_validation':verify,'checkpoint':str(ART/'best.pt'),'checkpoint_sha256':sha(ART/'best.pt'),'scope':scope,'frozen_hash_before':frozen_before,'frozen_hash_after':state_hash(model,True),'train_frames':len(train),'validation_frames':len(val),'seed':cfg['seed'],'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-start,'epoch0_selected':best_epoch==0,'protocol_sha256':sha(OUT/'detector_protocol.json'),'adapted_hash':state_hash(model,False)})
    print(json.dumps({'completed':True,'selected_epoch':best_epoch,'validation_loss':best}),flush=True)


if __name__=='__main__':main()
