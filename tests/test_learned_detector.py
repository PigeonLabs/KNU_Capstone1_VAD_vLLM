from types import SimpleNamespace
import numpy as np
import pytest
import torch
from transformers import BatchEncoding
from ipad_vad.learned_detector import scene_batches, forward_loss


def test_batches_preserve_each_frame_and_do_not_mix_prompt_spaces():
    rows=[{'id':f'{s}-{i}','scene':s} for s,n in [('R01',7),('R03',4),('R04',3)] for i in range(n)]
    a=scene_batches(rows,rng=np.random.default_rng(42));b=scene_batches(rows,rng=np.random.default_rng(42))
    assert a==b
    assert sorted(r['id'] for batch in a for r in batch)==sorted(r['id'] for r in rows)
    assert all(len(batch)<=2 and len({r['scene'] for r in batch})==1 for batch in a)
    with pytest.raises(ValueError):scene_batches(rows,batch_size=3)


def test_decoder_matching_loss_ignores_detached_encoder_and_respects_roles():
    logits=torch.full((1,2,256),-10.);logits[0,0,1]=10.;logits[0,1,3]=10.;logits.requires_grad_()
    boxes=torch.tensor([[[.2,.3,.1,.2],[.7,.6,.2,.3]]],requires_grad=True)
    options=SimpleNamespace(two_stage=True,auxiliary_loss=False,class_cost=1.,bbox_cost=5.,giou_cost=2.,focal_alpha=.25,bbox_loss_coefficient=5.,giou_loss_coefficient=2.)
    class Model:
        config=options;device=torch.device('cpu')
        def __call__(self,**kwargs):return SimpleNamespace(logits=logits,pred_boxes=boxes,encoder_logits=torch.full_like(logits,float('nan')))
    inputs=BatchEncoding({'input_ids':torch.tensor([[101,42,1012,43,1012,102]]),'attention_mask':torch.ones((1,6),dtype=torch.long)})
    targets=[{'class_labels':torch.tensor([0,1]),'boxes':boxes[0].detach().clone()}]
    output=forward_loss(Model(),inputs,targets)
    assert output.loss<.001 and torch.isfinite(output.loss)
    output.loss.backward();assert logits.grad is not None and boxes.grad is not None
    assert options.two_stage and not any(k.endswith('_enc') for k in output.loss_dict)
    targets[0]['class_labels']=torch.tensor([1,0]);wrong=forward_loss(Model(),inputs,targets)
    assert wrong.loss>output.loss+1.
