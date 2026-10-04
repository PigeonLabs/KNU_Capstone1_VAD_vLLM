import copy
import numpy as np
import pytest
from sklearn.cluster import KMeans
from ipad_vad.asinh_phase import AsinhConfirmedAnchorPhase
from ipad_vad.confirmed_anchor import ConfirmedAnchorPhase
from test_relational_phase import fixture


def setup():
    d=fixture();d['crop_features']=np.tile([1.,0.],(len(d['roles']),1));m=AsinhConfirmedAnchorPhase(np.eye(2),anchor_role=0,target_role=1,reset_on_track_change=True);m.fit_gates([d])
    x,v,_=m.descriptors(d);m.location=np.median(x[v],axis=0);q1,q3=np.quantile(x[v],[.25,.75],axis=0);m.scale=np.maximum(q3-q1,.05)
    return d,m


def test_compression_fit_and_frozen_scaler_selection(monkeypatch):
    d,m=setup();before=copy.deepcopy(d);stats=[m.location.copy(),m.scale.copy(),copy.deepcopy(m.area_upper)];x,v,c=m.descriptors(d)
    def forbidden(_):raise AssertionError('Gate fit forbidden')
    monkeypatch.setattr(m,'fit_gates',forbidden);m.fit([d]);u=np.arcsinh((x[v]-stats[0])/stats[1]);km=KMeans(n_clusters=4,random_state=42,n_init=10).fit(u);time=d['indices'][v]/(int(d['frame_count'])-1);order=np.argsort([np.median(time[km.labels_==i]) for i in range(4)],kind='stable')
    np.testing.assert_array_equal(m.centers,km.cluster_centers_[order]);np.testing.assert_array_equal(m.location,stats[0]);np.testing.assert_array_equal(m.scale,stats[1]);assert m.area_upper==stats[2]
    for a,b in zip(m.transform(d)[1:],(v,c,x)):np.testing.assert_array_equal(a,b)
    for k in d:np.testing.assert_array_equal(d[k],before[k])
    z=np.array([-1e8,-1.,0.,1.,1e8]);probe=np.tile(m.location,(len(z),1));probe[:,0]+=z*m.scale[0]
    np.testing.assert_allclose(m.phase_coordinates(probe)[:,0],np.arcsinh(z));assert np.abs(np.arcsinh(z[-1]))<20


def test_transform_uses_compressed_not_linear_distance_and_holds_invalid_phase():
    d,m=setup();m.location=np.zeros(6);m.scale=np.ones(6);m.centers=np.zeros((4,6));m.centers[:,0]=[0,2,4,6]
    x=np.zeros((3,6));x[:,0]=[10,0,100];v=np.array([True,False,True]);c=np.zeros((3,2),int);m.descriptors=lambda _: (x,v,c)
    # asinh(10)=2.998.. picks2, not linear10 picking6; missing holds prior.
    np.testing.assert_array_equal(m.transform({})[0],[1,1,3])


def test_serialization_restores_transform_and_predictions(tmp_path):
    d,m=setup();m.fit([d]);path=tmp_path/'phase.npz';m.save_phase(path);other=AsinhConfirmedAnchorPhase(np.eye(2),anchor_role=0,target_role=1,reset_on_track_change=True).load_phase(path)
    for a,b in zip(m.transform(d),other.transform(d)):np.testing.assert_array_equal(a,b)
    with np.load(path) as f:bad=dict(f)
    bad['coordinate_transform']=np.array('linear');np.savez(path,**bad)
    with pytest.raises(ValueError,match='asinh'):other.load_phase(path)


def test_prefix_future_and_inference_time_independent():
    d,m=setup();m.fit([d]);full=m.transform(d)
    for end in [1,21,40,79]:
        prefix={k:v[:end] if k=='indices' else v if k=='frame_count' else v[d['object_frames']<end] for k,v in d.items()}
        for a,b in zip(m.transform(prefix),full):np.testing.assert_array_equal(a,b[:end])
    other=copy.deepcopy(d);other['boxes'][80:]=[0,0,1,1]
    for a,b in zip(m.transform(other),full):np.testing.assert_array_equal(a[:40],b[:40])
    other=copy.deepcopy(d);other['indices']=other['indices']*7+1000;other['frame_count']=np.array(99999)
    for a,b in zip(m.transform(other),full):np.testing.assert_array_equal(a,b)


def test_refused_fit_keeps_frozen_model():
    d,m=setup();m.fit([d]);centers=m.centers.copy();d['roles'][:]=99
    with pytest.raises(ValueError,match='support'):m.fit([d])
    np.testing.assert_array_equal(m.centers,centers)
    with pytest.raises(ValueError,match='caches'):m.fit([])
    m.scale[0]=0
    with pytest.raises(ValueError,match='positive scale'):m.phase_coordinates(np.zeros((1,6)))
