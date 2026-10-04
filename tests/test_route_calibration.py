import numpy as np
import pytest
from ipad_vad.route_calibration import RouteCalibration,leave_one_video_out
from ipad_vad.scoring import empirical_percentile
from test_observed_appearance import setup


def test_observation_and_distinct_video_support_both_required():
    r=RouteCalibration(50,2)
    rows=[(0,np.zeros(30,int),np.arange(30.),'a'),(0,np.zeros(30,int),np.arange(30.),'a')]
    r.fit(rows);assert not r.references and r.support[0,0]['observations']==60
    rows[1]=(0,np.zeros(30,int),np.arange(30.),'b');r.fit(rows);assert (0,0) in r.references
    r.fit([(0,np.zeros(20,int),np.arange(20.),s) for s in ['a','b']]);assert not r.references
    np.testing.assert_array_equal(r.score(0,np.array([0,1]),np.array([1.,5.]),np.arange(10.)),empirical_percentile(np.arange(10.),[1.,5.]))


def test_route_reference_constant_ties_and_unseen_route_fallback():
    r=RouteCalibration(2,2);r.fit([(0,np.zeros(2,int),np.zeros(2),s) for s in ['a','b']])
    np.testing.assert_array_equal(r.score(0,np.array([0,0,1]),np.array([0.,1.,1.]),np.array([0.,1.,2.])),[.5,1.,.5])
    r.fit([(0,np.ones(2,int),np.ones(2),s) for s in ['a','b']]);assert (0,0) not in r.references
    with pytest.raises(ValueError):RouteCalibration(0,2)


def test_actual_pooled_route_for_observed_but_unsupported_phase():
    m,d=setup();d['relation_valid'][0]=False;m.fit([d])
    routes=m.appearance_routes(d,0,np.arange(40))
    assert routes[1]==0 and d['relation_valid'][1]  # phase 0 has only nine FIT rows
    assert routes[20]==1 and routes[39]==0
    assert np.all(m.appearance_routes(d,99,np.arange(40))==0)


def test_calibration_and_inference_route_references_and_duplicate_ids():
    m,d=setup();m.cfg['appearance_route_calibration']={'minimum_observations':30,'minimum_videos':2};m.fit([d])
    cal=[{**d,'sequence_id':s} for s in ['a','b']];m.calibrate(cal)
    for role,frames,r in m.raw(d)[0]:
        routes=m.appearance_routes(d,role,frames);scored=m.calibrate_appearance(d,role,frames,r)
        for route in [0,1]:
            mask=routes==route;reference=m.route_calibration.references[role,route]
            np.testing.assert_array_equal(reference,np.tile(r[mask],2));np.testing.assert_array_equal(scored[mask],empirical_percentile(reference,r[mask]))
    assert m.threshold==np.quantile(np.concatenate([m.score(c)['combined'] for c in cal]),.99,method='higher')
    with pytest.raises(ValueError):m.calibrate([cal[0],cal[0]])


def test_holdout_excluded_from_refs_and_threshold():
    m,d=setup();m.cfg['appearance_route_calibration']={'minimum_observations':30,'minimum_videos':2};m.fit([d])
    caches={s:{**d,'sequence_id':s} for s in ['a','b','held']}
    held,used=leave_one_video_out(list(caches))[-1];assert held=='held' and set(used)=={'a','b'}
    m.calibrate([caches[s] for s in used]);q=m.threshold;refs={k:v.copy() for k,v in m.route_calibration.references.items()}
    caches[held]={**d,'sequence_id':held,'global_features':d['global_features']+999}
    m.calibrate([caches[s] for s in used]);assert m.threshold==q
    for k,v in refs.items():np.testing.assert_array_equal(v,m.route_calibration.references[k]);assert held not in m.route_calibration.support[k]['videos']
    with pytest.raises(ValueError):leave_one_video_out(['a','a'])
