import copy
import numpy as np
import pytest
from test_observed_appearance import setup
from ipad_vad.rank_control import constrain_phase_ranks


def test_prefix_basis_preserves_mean_pool_and_increases_raw_residual():
    base,d=setup('phase');base.fit([d]);m,_=setup('phase');m.cfg['appearance_phase_ranks']={f'{r}:{p}':1 for r,p in base.spaces if p>=0};m.fit([d])
    for key,s in m.spaces.items():
        old=base.spaces[key];np.testing.assert_array_equal(s.mean,old.mean);assert s.n==old.n
        np.testing.assert_array_equal(s.basis,old.basis if key[1]<0 else old.basis[:1])
    for (_,_,a),(_,_,b) in zip(base.raw(d)[0],m.raw(d)[0]):assert np.all(b>=a-1e-12)
    np.testing.assert_array_equal(base.transition,m.transition)
    m.calibrate([d]);assert m.threshold==np.quantile(m.score(d)['combined'],.99,method='higher')


def test_same_rank_identity_and_no_partial_mutation():
    m,d=setup();m.fit([d]);before=copy.deepcopy(m.spaces);spec={f'{r}:{p}':s.rank for (r,p),s in m.spaces.items() if p>=0};constrain_phase_ranks(m.spaces,spec)
    for k,s in before.items():np.testing.assert_array_equal(s.basis,m.spaces[k].basis)
    spec['-1:0']=1;spec['0:1']=999
    with pytest.raises(ValueError):constrain_phase_ranks(m.spaces,spec)
    for k,s in before.items():np.testing.assert_array_equal(s.basis,m.spaces[k].basis)


@pytest.mark.parametrize('error',['missing','extra','pool','zero','negative','bool','float','too_large'])
def test_reject_incomplete_or_invalid_maps(error):
    m,d=setup();m.fit([d]);spec={f'{r}:{p}':s.rank for (r,p),s in m.spaces.items() if p>=0}
    if error=='missing':del spec['-1:0']
    elif error=='extra':spec['99:0']=1
    elif error=='pool':spec['-1:-1']=1
    else:spec['-1:0']={'zero':0,'negative':-1,'bool':True,'float':1.5,'too_large':100}[error]
    with pytest.raises(ValueError):constrain_phase_ranks(m.spaces,spec)
