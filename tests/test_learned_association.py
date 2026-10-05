import numpy as np
import torch
from ipad_vad.learned_association import AssociationTracker,ResidualMetric,pair_terms
from ipad_vad.tracking import Tracker

def test_no_support_is_exact_iou_even_empty_frames():
    rng=np.random.default_rng(4);a=Tracker();b=AssociationTracker({})
    for t in range(12):
        n=t%5;lo=rng.uniform(0,.7,(n,2));boxes=np.c_[lo,lo+.2];roles=rng.integers(0,3,n);features=rng.normal(size=(n,512));features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-9)
        np.testing.assert_array_equal(a.update(boxes,roles,t),b.update(boxes,roles,t,features,np.ones(n)))

def test_adapter_identity_then_receives_gradient():
    torch.manual_seed(42);m=ResidualMetric();x=torch.randn(16,512);y=x+.1*torch.randn_like(x)
    torch.testing.assert_close(m(x),torch.nn.functional.normalize(x,dim=-1));_,n,_=pair_terms(m,x,y);n.mean().backward();assert m.up.weight.grad.norm()>0

def test_reconnect_gate_age_role_and_no_hallucination():
    a=AssociationTracker({0:.8});x=np.array([[1.,0.]])
    assert a.update(np.array([[.1,.1,.2,.2]]),[0],0,x,[.9]).tolist()==[0]
    assert len(a.update(np.empty((0,4)),[],1,np.empty((0,2)),[]))==0
    assert a.update(np.array([[.21,.1,.31,.2]]),[0],2,x,[.9]).tolist()==[0]
    assert len(a.edges)==1
    assert a.update(np.array([[.32,.1,.42,.2]]),[1],3,x,[.9]).tolist()==[1]
    assert a.update(np.array([[.43,.1,.53,.2]]),[0],5,x,[.9]).tolist()==[2]

def test_ambiguous_candidates_are_not_forced():
    a=AssociationTracker({0:.8});a.update(np.array([[.1,.1,.2,.2]]),[0],0,np.array([[1.,0.]]),[1.])
    ids=a.update(np.array([[.21,.1,.31,.2],[.21,.21,.31,.31]]),[0,0],1,np.array([[1.,0.],[1.,0.]]),[1.,1.])
    assert 0 not in ids and len(set(ids))==2 and not a.edges

def test_iou_priority_and_unsupported_gate():
    for gates in [{0:1.01},{0:.8}]:
        a=AssociationTracker(gates);a.update(np.array([[.1,.1,.3,.3]]),[0],0,np.array([[1.,0.]]),[1.])
        ids=a.update(np.array([[.11,.1,.31,.3],[.32,.1,.52,.3]]),[0,0],1,np.array([[-1.,0.],[1.,0.]]),[1.,1.])
        assert ids.tolist()==[0,1] and not a.edges
