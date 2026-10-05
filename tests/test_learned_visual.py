import copy
import torch
import pytest
from torch import nn
from ipad_vad.learned_visual import LoRALinear, configure_adaptation, representation_loss


def test_lora_zero_init_gradient_update_and_restore():
    torch.manual_seed(42);base=nn.Linear(6,4);original=copy.deepcopy(base);m=LoRALinear(base,2,4)
    x=torch.randn(5,6);expected=original(x).detach();torch.testing.assert_close(m(x),expected,rtol=0,atol=0)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
    m(x).square().mean().backward();assert m.lora_b.grad.abs().sum()>0 and m.base.weight.grad is None
    opt.step();assert not torch.equal(m(x),expected)
    for a,b in zip(original.parameters(),m.base.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    restored=LoRALinear(copy.deepcopy(original),2,4);restored.load_state_dict(m.state_dict());torch.testing.assert_close(m(x),restored(x),rtol=0,atol=0)


def test_full_and_frozen_parameter_scope_and_buffers():
    m=nn.Sequential(nn.Linear(3,3),nn.BatchNorm1d(3));before=m[1].running_mean.clone()
    r=configure_adaptation(m,'full');assert r['trainable_parameters']==r['total_parameters']
    m(torch.randn(4,3));torch.testing.assert_close(before,m[1].running_mean,rtol=0,atol=0)
    assert configure_adaptation(m,'frozen')['trainable_parameters']==0
    with pytest.raises(ValueError):configure_adaptation(m,'lora')


def test_loss_uses_same_scene_other_video_and_teacher_has_no_gradient():
    torch.manual_seed(0);z=torch.randn(4,5,requires_grad=True);other=torch.randn(4,5,requires_grad=True);t=torch.randn(4,5,requires_grad=True)
    scenes=torch.tensor([0,0,1,1]);videos=torch.tensor([0,1,2,3]);loss,_=representation_loss(z,other,t,scenes,videos);loss.backward()
    assert torch.isfinite(loss) and z.grad.abs().sum()>0 and t.grad is None
    # The loss decomposes into scene-local losses: cross-scene negatives excluded.
    a=representation_loss(z[:2],other[:2],t[:2],scenes[:2],videos[:2])[0]
    b=representation_loss(z[2:],other[2:],t[2:],scenes[2:],videos[2:])[0]
    torch.testing.assert_close(loss,(a+b)/2)
    with pytest.raises(ValueError):representation_loss(z,other,t,scenes,torch.zeros(4,dtype=torch.long))


def test_balanced_sampler_repeats_deterministically_and_excludes_calibration():
    from ipad_vad.representation_data import BalancedBatches,AdaptationDataset
    rows=[dict(scene=f'R{s:02}',sequence=str(v),kind=k,partition=p) for s in range(1,5) for v in range(4) for k in ['global','crop'] for p in ['representation_train','representation_validation','normal_calibration']]
    sampler=BalancedBatches(rows,'representation_train',128,64,42)
    first=list(sampler);assert first==list(sampler)
    for batch in first:
        selected=[rows[i] for i,_ in batch]
        assert all(r['partition']=='representation_train' for r in selected)
        for s in range(1,5):
            group=[r for r in selected if r['scene']==f'R{s:02}']
            assert len(group)==16 and len({r['sequence'] for r in group})==4
            assert sum(r['kind']=='global' for r in group)==8
    sampler.epoch=1;assert list(sampler)!=first
    with pytest.raises(ValueError):AdaptationDataset(rows,None,None)
