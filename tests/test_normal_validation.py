import numpy as np
import pytest
from ipad_vad.normal_validation import leave_one_video_out,calibrate_holdout,fit_arrays,array_fingerprint,reference_arrays
from test_process_calibration import model
from test_baseline import fixture


def test_video_folds_exclude_heldout_and_fixed_fit_set():
    folds=leave_one_video_out(['a','b'],['c','d','e','f'])
    assert [f['held_out'] for f in folds]==['c','d','e','f']
    for fold in folds:
        assert fold['held_out'] not in fold['calibration'] and len(fold['calibration'])==3
        assert not set(fold['calibration'])&{'a','b'}
    with pytest.raises(ValueError):leave_one_video_out(['a'],['a','b'])
    with pytest.raises(ValueError):leave_one_video_out(['a'],['b','b'])


def test_heldout_changes_cannot_change_any_reference_or_threshold():
    fitted=model();fitted.fit([fixture(0),fixture(1)]);before=array_fingerprint(fit_arrays(fitted))
    caches={s:fixture(seed) for seed,s in enumerate(['c','d','e','f'],start=2)}
    a,cal,out=calibrate_holdout(fitted,caches,['c','d','e'],'f')
    changed={s:{k:v.copy() for k,v in d.items()} for s,d in caches.items()}
    changed['f']['global_features']+=1e6;changed['f']['crop_features']+=1e6;changed['f']['phases'][:]=2
    b,_,perturbed=calibrate_holdout(fitted,changed,['c','d','e'],'f')
    for key,value in reference_arrays(a).items():np.testing.assert_array_equal(value,reference_arrays(b)[key])
    assert a.threshold==np.quantile(np.concatenate([r['combined'] for r in cal]),.99,method='higher')
    assert len(a.process_reference)==120
    assert array_fingerprint(fit_arrays(fitted))==before and not hasattr(fitted,'process_reference')
    assert not np.array_equal(out['visual'],perturbed['visual'])
    with pytest.raises(ValueError):calibrate_holdout(fitted,caches,['c','d','f'],'f')
