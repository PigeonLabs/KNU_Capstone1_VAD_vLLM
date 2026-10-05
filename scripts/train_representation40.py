"""Matched-budget shared MobileCLIP2 adaptation using only normal train/val views."""
import argparse,copy,json,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ipad_vad.learned_visual import load_mobile_visual,configure_adaptation,representation_loss
from ipad_vad.representation_data import read_views,AdaptationDataset,BalancedBatches
from prepare_representation40 import sha,write,ART,OUT


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def access_guard(rows):
    allowed={str(Path(r['path']).resolve()) for r in rows};opened=set()
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]);s=str(p.resolve())
        if '/test_label/' in s or '/testing/frames/' in s or p.name.startswith('testing_'):raise RuntimeError('Test access forbidden during representation training: '+s)
        if '/IPAD_dataset/' in s and p.suffix.lower() in ['.jpg','.png','.jpeg']:
            if s not in allowed:raise RuntimeError('Image outside representation train/validation split: '+s)
            opened.add(s)
    sys.addaudithook(hook);return opened


def teacher_features(rows,manifest):
    buckets={};features=[];hashes={}
    extraction=json.loads((OUT/'B_normal_extraction.json').read_text());expected={(r['scene'],r['sequence']):r['sha256'] for r in extraction['sequences']}
    for row in rows:
        key=(row['scene'],row['sequence'])
        if key not in buckets:
            assert row['sequence'] not in manifest['subsplits'][row['scene']]['normal_calibration']
            p=ART/'B/features'/row['scene']/f'training_{row["sequence"]}.npz';assert sha(p)==expected[key]
            with np.load(p,allow_pickle=False) as f:buckets[key]={k:f[k] for k in ['global_features','crop_features']}
            hashes[str(p)]=expected[key]
        kind='global_features' if row['kind']=='global' else 'crop_features';idx=row['sample_index'] if row['kind']=='global' else row['detection_index'];features.append(buckets[key][kind][idx])
    return np.stack(features),hashes


def pass_batches(model,loader,cfg,optimizer=None,step=0,total_steps=1):
    totals={k:0. for k in ['loss','teacher_cosine_loss','contrastive_loss']};count=0;vectors=[]
    model.eval() # Fixed normalization statistics in both arms; gradients remain enabled.
    for batch_index,(a,b,teacher,scenes,videos) in enumerate(loader):
        a,b,teacher,scenes,videos=[x.cuda(non_blocking=True) for x in [a,b,teacher,scenes,videos]]
        if optimizer:
            warmup=cfg['samples_per_epoch']//cfg['batch_size']*cfg['warmup_epochs']
            factor=(step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
            for group in optimizer.param_groups:group['lr']=cfg['learning_rate']*factor
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None),torch.autocast('cuda',dtype=torch.bfloat16):
            z=model(torch.cat([a,b]));z1,z2=z.chunk(2)
            loss,parts=representation_loss(z1,z2,teacher,scenes,videos,cfg['contrastive_temperature'],cfg['contrastive_weight'])
        if not torch.isfinite(loss):raise RuntimeError('Non-finite representation loss')
        if optimizer:
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['gradient_clip_norm'])
            if not torch.isfinite(norm):raise RuntimeError('Non-finite gradient norm')
            optimizer.step();step+=1
            if step%16==0:print(json.dumps({'optimizer_step':step,'batch_loss':float(loss.detach())}),flush=True)
        else:vectors.append(torch.nn.functional.normalize(z1.float(),dim=-1).detach().cpu())
        n=len(a);count+=n;totals['loss']+=float(loss.detach())*n
        for key,value in parts.items():totals[key]+=float(value)*n
    result={k:v/count for k,v in totals.items()};result['samples']=count
    if vectors:
        v=torch.cat(vectors).numpy();centered=v-v.mean(0);sv=np.linalg.svd(centered,compute_uv=False);energy=sv**2
        prob=energy/energy.sum() if energy.sum()>0 else energy
        result['feature_mean_std']=float(v.std(0).mean());result['effective_rank']=float(np.exp(-np.sum(prob[prob>0]*np.log(prob[prob>0])))) if energy.sum()>0 else 0.
    return result,step


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['C','D'],required=True);p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    cfg=json.loads(Path('configs/experiment40_representation.json').read_text());assert args.seed in cfg['seeds'];torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent CPU replacement')
    protocol=json.loads((OUT/'representation_protocol.json').read_text())
    for path,digest in protocol['file_sha256'].items():assert sha(path)==digest,path
    manifest=json.loads((OUT/'data_manifest.json').read_text());assert sha(manifest['view_manifest'])==manifest['view_manifest_sha256']
    for name,digest in manifest['model']['sha256'].items():assert sha(Path(manifest['model']['local_path'])/name)==digest,name
    rows=read_views(manifest['view_manifest'],['representation_train','representation_validation']);opened=access_guard(rows);teacher,teacher_hashes=teacher_features(rows,manifest)
    seed_all(args.seed);model,_=load_mobile_visual(manifest['model']);scope=configure_adaptation(model,'lora' if args.arm=='C' else 'full',cfg['lora_rank'],cfg['lora_alpha']);model=model.cuda()
    run=f'{args.arm}_s{args.seed}';dest=ART/run;dest.mkdir(parents=True,exist_ok=True)
    if (OUT/f'{run}_training.json').exists() or (dest/'best.pt').exists():raise RuntimeError('Existing run found; never restart/overwrite an ambiguous training run')
    frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad};buffers={n:b.detach().cpu().clone() for n,b in model.named_buffers()};initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    dataset=AdaptationDataset(rows,teacher,cfg['augmentation']);train_batches=BalancedBatches(rows,'representation_train',cfg['samples_per_epoch'],cfg['batch_size'],args.seed);val_batches=BalancedBatches(rows,'representation_validation',cfg['validation_samples'],cfg['batch_size'],9040)
    # Workers receive only the filtered train/validation Dataset; no test/calibration paths.
    loader_kwargs=dict(num_workers=cfg['workers'],pin_memory=True,persistent_workers=cfg['workers']>0)
    if cfg['workers']:loader_kwargs.update(multiprocessing_context='spawn',prefetch_factor=cfg['prefetch_factor'])
    training=DataLoader(dataset,batch_sampler=train_batches,**loader_kwargs);validation=DataLoader(dataset,batch_sampler=val_batches,num_workers=cfg['validation_workers'],pin_memory=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    protocol_files=['configs/experiment40_representation.json','scripts/train_representation40.py','scripts/prepare_representation40.py','src/ipad_vad/learned_visual.py','src/ipad_vad/representation_data.py','results/experiment40/data_manifest.json','results/experiment40/B_normal_extraction.json']
    protocol={'run':run,'seed':args.seed,'scope':scope,'file_sha256':{p:sha(p) for p in protocol_files},'teacher_normal_feature_sha256':teacher_hashes,'model':manifest['model'],'normal_only':True,'calibration_and_test_excluded':True}
    write(OUT/f'{run}_pretrain_protocol.json',protocol);start=time.perf_counter();torch.cuda.reset_peak_memory_stats();history=[];best=float('inf');best_epoch=None;step=0
    for epoch in range(cfg['epochs']+1):
        train=None
        if epoch:
            train_batches.epoch=epoch
            train,step=pass_batches(model,training,cfg,optimizer,step,cfg['epochs']*len(train_batches))
        val,_=pass_batches(model,validation,cfg)
        improved=val['loss']<best
        if improved:
            best=val['loss'];best_epoch=epoch
            torch.save({'visual':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch,'run':run,'seed':args.seed,'scope':scope,'protocol_sha256':sha(OUT/f'{run}_pretrain_protocol.json')},dest/'best.pt')
        row={'epoch':epoch,'step':step,'train':train,'validation':val,'selected_so_far':best_epoch,'elapsed_seconds':time.perf_counter()-start};history.append(row);write(OUT/f'{run}_history.json',history);print(json.dumps(dict(run=run,**row)),flush=True)
    for name,value in frozen.items():torch.testing.assert_close(dict(model.named_parameters())[name].detach().cpu(),value,rtol=0,atol=0)
    for name,value in buffers.items():torch.testing.assert_close(dict(model.named_buffers())[name].detach().cpu(),value,rtol=0,atol=0)
    changes={n:float(torch.linalg.vector_norm(p.detach().cpu()-initial[n])) for n,p in model.named_parameters() if p.requires_grad};assert any(v>0 for v in changes.values())
    ckpt=torch.load(dest/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ckpt['visual'],strict=True);restored,_=pass_batches(model,validation,cfg)
    np.testing.assert_allclose(restored['loss'],history[best_epoch]['validation']['loss'],rtol=1e-5,atol=1e-6)
    torch.cuda.synchronize();write(OUT/f'{run}_training.json',{'run':run,'seed':args.seed,'scope':scope,'selected_epoch':best_epoch,'selected_training_checkpoint':best_epoch>0,'epochs_run':cfg['epochs'],'optimizer_steps':step,'checkpoint':str(dest/'best.pt'),'checkpoint_sha256':sha(dest/'best.pt'),'elapsed_seconds':time.perf_counter()-start,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'frozen_parameters_exact':True,'normalization_buffers_exact':True,'parameter_update_norms':changes,'restored_validation':restored,'normal_only':True,'calibration_and_test_excluded':True,'teacher_feature_files':len(teacher_hashes),'worker_input_image_allowlist_count':len({r['path'] for r in rows}),'main_process_opened_images':len(opened),'history_path':str(OUT/f'{run}_history.json')})

if __name__=='__main__':main()
