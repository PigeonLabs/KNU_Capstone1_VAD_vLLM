import numpy as np
from ipad_vad.relational_phase import RelationalPhase


def fixture():
    n=80;boxes=[]
    for i in range(n):
        x=.4+.2*i/n;dx=[-.18,-.02,.04,.18][i//20]
        boxes.extend([[x-.1,.3,x+.1,.6],[x+dx-.025,.4,x+dx+.025,.47]])
    return {'indices':np.arange(n)*4,'frame_count':np.array(n*4),'boxes':np.array(boxes),
            'roles':np.tile([0,1],n),'tracks':np.tile([0,1],n),'object_frames':np.repeat(np.arange(n),2),'confidence':np.ones(n*2)}


def test_states_are_causal_and_ignore_test_time_position():
    d=fixture();m=RelationalPhase();m.fit([d]);before=m.transform(d)[0]
    changed={k:v.copy() for k,v in d.items()};changed['boxes'][80:]=[.1,.1,.2,.2]
    np.testing.assert_array_equal(before[:40],m.transform(changed)[0][:40])
    shifted={k:v.copy() for k,v in d.items()};shifted['indices']=shifted['indices']*13+900;shifted['frame_count']=np.array(100000)
    np.testing.assert_array_equal(before,m.transform(shifted)[0])
    assert len(np.unique(before))==4


def test_background_box_is_not_selected_and_missing_relation_is_marked():
    d=fixture();m=RelationalPhase();m.fit([d]);d['boxes']=np.vstack([d['boxes'],[0,0,1,1]])
    for k,v in [('roles',1),('tracks',99),('object_frames',0),('confidence',100)]:d[k]=np.r_[d[k],v]
    phases,valid,chosen,_=m.transform(d);assert chosen[0,1]==1
    d['roles'][d['object_frames']==10]=2
    p,v,_,x=m.transform(d);assert not v[10] and p[10]==p[9]
    assert np.all(x[10]==0), 'Invalid sentinel must be separately masked'
    one=RelationalPhase(smoothing=1);one.area_upper=m.area_upper
    expected=one.descriptors(d)[0][11]
    np.testing.assert_allclose(x[11],expected), 'Missing interval must clear smoothing history'
