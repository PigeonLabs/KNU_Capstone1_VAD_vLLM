"""GPU architecture/gradient smoke; synthetic inputs, no scientific training run."""
import copy,json,time
from pathlib import Path
import torch
from ipad_vad.learned_visual import load_mobile_visual,configure_adaptation,representation_loss
from prepare_representation40 import OUT,write


def main():
    torch.set_num_threads(4);torch.manual_seed(40)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    record=json.loads(Path('artifacts/experiment40/mobileclip_model.json').read_text());base,_=load_mobile_visual(record);base=base.cuda().eval();base.requires_grad_(False)
    images=torch.rand(4,3,256,256,device='cuda')
    with torch.no_grad():target=base(images).detach()
    with torch.no_grad():
        clone=copy.deepcopy(base);clone_error=float((clone(images)-target).abs().max());repeat_error=float((base(images)-target).abs().max());del clone
    rows=[]
    for arm,mode in [('C','lora'),('D','full')]:
        model=copy.deepcopy(base);scope=configure_adaptation(model,mode);before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};buffers={n:p.detach().clone() for n,p in model.named_buffers()}
        with torch.no_grad():
            initial_output=model(images);initial_error=float((initial_output-target).abs().max())
            torch.testing.assert_close(initial_output,target,rtol=1e-5,atol=1e-6)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-5)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            a=model(images);b=model((images*.97).clamp(0,1));loss,_=representation_loss(a,b,target,torch.zeros(4,device='cuda',dtype=torch.long),torch.arange(4,device='cuda'))
        loss.backward();assert torch.isfinite(loss)
        names=[n for n,p in model.named_parameters() if p.grad is not None and p.grad.abs().sum()>0];assert names
        optimizer.step();changed=[n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(p,before[n])];assert changed
        for name,value in buffers.items():torch.testing.assert_close(dict(model.named_buffers())[name],value,rtol=0,atol=0)
        rows.append({'arm':arm,'scope':scope,'zero_initialization_output_within_tolerance':True,'initial_output_max_absolute_difference':initial_error,'copy_without_adaptation_max_absolute_difference':clone_error,'repeat_max_absolute_difference':repeat_error,'output_rtol':1e-5,'output_atol':1e-6,'finite_loss':float(loss.detach()),'nonzero_gradient_parameters':len(names),'changed_parameter_tensors':len(changed),'batchnorm_buffers_exact':True})
        del model,optimizer,before,buffers
    write(OUT/'gpu_smoke.json',{'synthetic_smoke_only':True,'no_checkpoint_used_for_experiment':True,'numerical_preflight_note':'Bitwise end-to-end equality failed at max 3.22e-5 under default TF32, and 8.94e-8 with TF32 disabled. Linear zero-init is unit-tested exactly; GPU tower is checked at rtol1e-5/atol1e-6, with no-adaptation copy and repeat controls.','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'runs':rows});print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
