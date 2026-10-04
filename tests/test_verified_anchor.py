import numpy as np
from ipad_vad.verified_anchor import VerifiedAnchorPhase
from test_relational_phase import fixture


def prepared():
    d=fixture();d['crop_features']=np.tile([[1.,0.],[0.,1.]],(80,1));return d


def test_gate_changes_only_anchor_role_view_and_rejects_ties():
    d=prepared();m=VerifiedAnchorPhase(np.eye(2));m.fit([d]);original={k:v.copy() for k,v in d.items()}
    d['crop_features'][20]=[0,1];d['crop_features'][22]=[1,1]
    view=m.phase_view(d)
    assert view['roles'][20]==-999 and view['roles'][22]==-999
    np.testing.assert_array_equal(view['roles'][1::2],d['roles'][1::2])
    for k in d:
        if k!='crop_features':np.testing.assert_array_equal(d[k],original[k])
    _,valid,chosen,_=m.transform(d)
    assert not valid[10] and not valid[11] and chosen[10,0]==-1


def test_gate_is_causal_and_cosine_scale_invariant():
    d=prepared();m=VerifiedAnchorPhase(np.eye(2));m.fit([d]);before=m.transform(d)
    changed={k:v.copy() for k,v in d.items()};changed['crop_features'][80:]=[0,1]
    after=m.transform(changed)
    for a,b in zip(before,after):np.testing.assert_array_equal(a[:40],b[:40])
    scaled={k:v.copy() for k,v in d.items()};scaled['crop_features']*=9
    np.testing.assert_allclose(m.margins(d),m.margins(scaled))
    np.testing.assert_array_equal(before[0],m.transform(scaled)[0])
