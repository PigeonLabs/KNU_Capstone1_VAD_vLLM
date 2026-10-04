import copy
import numpy as np
import pytest
from ipad_vad.relational_phase import RelationalPhase
from test_relational_phase import fixture


def test_reset_is_identity_bounded_and_causal():
    d=fixture();control=RelationalPhase();control.fit([d]);reset=copy.deepcopy(control);reset.reset_on_track_change=True
    for a,b in zip(control.transform(d),reset.transform(d)):np.testing.assert_array_equal(a,b)
    # Anchor then target replacement, followed by missing data and reacquisition.
    d['tracks'][d['object_frames']>=7]+=10
    d['tracks'][(d['object_frames']>=14)&(d['roles']==1)]+=10
    d['roles'][d['object_frames']==20]=99
    p,v,c,x=reset.transform(d);old=control.transform(d)
    np.testing.assert_array_equal(v,old[1]);np.testing.assert_array_equal(c,old[2])
    raw_model=copy.deepcopy(reset);raw_model.smoothing=1;raw=raw_model.descriptors(d)[0]
    history=[];prior=None
    for i in range(len(v)):
        if not v[i]:
            history=[];prior=None;assert p[i]==p[i-1];continue
        pair=tuple(d['tracks'][c[i]])
        if prior!=pair:history=[]
        history=(history+[raw[i]])[-3:];prior=pair
        np.testing.assert_array_equal(x[i],np.mean(history,axis=0))
    for i in [0,7,14,21]:np.testing.assert_array_equal(x[i],raw[i])
    for end in [1,7,8,14,15,20,21,22,40]:
        use=d['object_frames']<end
        prefix={k:(a[:end] if k=='indices' else a[use] if k in ['boxes','roles','tracks','object_frames','confidence'] else a) for k,a in d.items()}
        for a,b in zip(reset.transform(prefix),(p[:end],v[:end],c[:end],x[:end])):np.testing.assert_array_equal(a,b)
    assert np.any(x!=old[3])


def test_track_reset_rejects_ambiguous_configuration():
    with pytest.raises(ValueError):RelationalPhase(reset_on_track_change='false')
