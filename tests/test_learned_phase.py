import copy
import numpy as np
import torch
from ipad_vad.learned_phase import PhaseHead,causal_visual_input,causal_hold

def sample(n=6):
    rng=np.random.default_rng(1);return {'global_features':rng.normal(size=(n,512)).astype('float32'),'object_frames':np.arange(n),'roles':np.zeros(n,int),'confidence':np.ones(n),'crop_features':rng.normal(size=(n,512)).astype('float32'),'phases':np.arange(n),'indices':np.arange(n)*4,'frame_count':n*4}

def test_feature_inputs_are_causal_and_ignore_labels_time():
    d=sample();x=causal_visual_input(d);a=copy.deepcopy(d);a['phases'][:]=-99;a['indices']*=100;a['frame_count']=10**9;np.testing.assert_array_equal(x,causal_visual_input(a));a['global_features'][3:]*=-1;a['crop_features'][3:]*=5;np.testing.assert_array_equal(x[:3],causal_visual_input(a)[:3]);assert not np.array_equal(x[3:],causal_visual_input(a)[3:])

def test_linear_adapter_same_initial_predictions_and_trainable_residual():
    torch.manual_seed(42);a=PhaseHead(4,adapt=False);torch.manual_seed(42);b=PhaseHead(4,adapt=True);x=torch.nn.functional.normalize(torch.randn(8,2048),dim=-1);torch.testing.assert_close(a(x)[0],b(x)[0]);loss=torch.nn.functional.cross_entropy(b(x)[0],torch.arange(8)%4);loss.backward();assert b.up.weight.grad.norm()>0 and b.classifier.weight.grad.norm()>0

def test_missing_emissions_hold_past_without_future():
    np.testing.assert_array_equal(causal_hold([3,2,1,0],[False,True,False,True]),[0,2,2,0])
