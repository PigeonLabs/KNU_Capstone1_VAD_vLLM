import numpy as np
import pytest
from test_process_calibration import model
from test_baseline import fixture


def test_fixed_fusions_preserve_branches_and_use_their_own_normal_threshold():
    normal=[fixture(0),fixture(1)];cal=[fixture(2)];test=fixture(3);outputs={};thresholds=[]
    for mode in ['mean','max','visual']:
        m=model();m.cfg['score_fusion']=mode;m.fit(normal);m.calibrate(cal)
        outputs[mode]=m.score(test);values=m.score(cal[0])['combined'];thresholds.append(m.threshold)
        assert m.threshold==np.quantile(values,.99,method='higher')
    a=outputs['mean'];b=outputs['max'];v=outputs['visual']
    for branch in ['visual','process']:
        np.testing.assert_array_equal(a[branch],b[branch]);np.testing.assert_array_equal(a[branch],v[branch])
    np.testing.assert_array_equal(b['combined'],np.maximum(a['visual'],a['process']))
    np.testing.assert_array_equal(v['combined'],a['visual'])
    assert np.all(b['combined']>=v['combined'])
    assert len(set(thresholds))>1,'Each fusion must be recalibrated, not inherit a threshold'


def test_mean_default_compatibility_and_unknown_fusion_fails():
    m=model();v=np.array([.2,.9]);p=np.array([.8,.1])
    np.testing.assert_array_equal(m.fuse(v,p),.5*v+.5*p)
    m.cfg['score_fusion']='unsupported'
    with pytest.raises(ValueError):m.fuse(v,p)
