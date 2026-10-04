"""Freeze and audit selected-pair continuity as transition fusion evidence."""
import argparse,copy,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.data import hold_scores
from ipad_vad.transition_evidence import observed_transition_mask,same_track_pair_transition_mask
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from experiment30_track_reset import arrays,feasibility,write,KEYS
from evaluate_route_holdout import load_cache
from audit_missing_age import load,sha
from audit_bank_dispatch import verify_references

OUT=Path('results/experiment31');ART=Path('artifacts/experiment31');ROOT=Path('artifacts/experiment30/reset/features/R04')
BEFORE=['30_reset_hold','30_reset_pool','30_reset_age'];AFTER=['31_hold','31_pool','31_age'];VARIANTS=BEFORE+AFTER;PAIRS=dict(zip(AFTER,BEFORE))
SPLIT=json.loads(Path('results/stage00/splits.json').read_text())['R04']
STATE_NAMES=['first','missing','reacquired','same_pair','anchor_changed','target_changed','both_changed']
PRESERVED=['visual','transition','transition_raw','dwell','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']


def config(e):return json.loads(Path(f'configs/experiment{e}.json').read_text())


def freeze(name,files):
    p=OUT/name
    if p.exists():verify_freeze(name)
    else:write(p,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'file_sha256':{str(p):sha(p) for p in files}})


def verify_freeze(name):
    for p,h in json.loads((OUT/name).read_text())['file_sha256'].items():assert sha(p)==h,p


def evidence(d):
    """Independent scalar metadata audit; no labels or scores used."""
    result=np.zeros(len(d['phases']),np.int8)
    for i in range(1,len(result)):
        if not d['relation_valid'][i]:result[i]=1
        elif not d['relation_valid'][i-1]:result[i]=2
        else:
            a,b=d['relation_detection_indices'][i-1],d['relation_detection_indices'][i]
            assert (a>=0).all() and (b>=0).all();assert (d['object_frames'][a]==i-1).all() and (d['object_frames'][b]==i).all()
            anchor=int(d['tracks'][a[0]])!=int(d['tracks'][b[0]]);target=int(d['tracks'][a[1]])!=int(d['tracks'][b[1]])
            result[i]=6 if anchor and target else 4 if anchor else 5 if target else 3
    np.testing.assert_array_equal(observed_transition_mask(d),result>=3);np.testing.assert_array_equal(same_track_pair_transition_mask(d),result==3)
    return result


def coverage(d,sequence,partition):
    state=evidence(d);dense=hold_scores(d['indices'],state,int(d['frame_count']));rows={}
    for i,name in enumerate(STATE_NAMES):
        matrix=np.zeros((4,4),int)
        use=(state[1:]==i);np.add.at(matrix,(d['phases'][:-1][use],d['phases'][1:][use]),1)
        rows[name]={'samples':int(np.sum(state==i)),'frames':int(np.sum(dense==i)),'edge_counts':matrix.tolist()}
    return {'sequence':sequence,'partition':partition,'states':rows}


def model_arrays(m):
    a=arrays(m);a['dwell_reference']=m.dwell.reference
    for state,v in m.dwell.durations.items():a[f'dwell_durations_state_{state}']=v
    return a


def array_checks(a,b,skip_threshold=False):
    for k,v in a.items():
        if skip_threshold and k=='threshold':continue
        np.testing.assert_array_equal(v,b[k],err_msg=k)


def fitted_models(fit):
    models={}
    for e in VARIANTS:
        cfg=config(e);m=LognormalDwellBaseline(cfg,load_process(cfg));m.fit(fit);models[e]=m
        source=load(f'artifacts/experiment{PAIRS.get(e,e)}/normal_model.npz')
        for (role,phase),s in m.spaces.items():
            for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(s,attr),source[f'{attr}_{role}_{phase}'])
        np.testing.assert_array_equal(m.transition,source['transition'])
        for (a,b),v in m.dwell.log_parameters.items():np.testing.assert_array_equal(v,source[f'dwell_log_parameters_{a}_{b}'])
        if e.endswith('age'):assert m.missing_age.tau==56
    return models


def paired_checks(models,caches):
    for e in AFTER:
        old,new=models[PAIRS[e]],models[e];array_checks(model_arrays(old),model_arrays(new),True)
        assert new.threshold<=old.threshold
        for d in caches:
            a,b=old.score(d),new.score(d);state=evidence(d)
            for k in PRESERVED:np.testing.assert_array_equal(a[k],b[k],err_msg=k)
            for (_,_,x),(_,_,y) in zip(a['objects'],b['objects']):np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(b['transition_valid'],state==3);np.testing.assert_array_equal(b['transition_gated'],np.where(state==3,a['transition'],0))
            np.testing.assert_array_equal(b['process'],np.maximum(b['transition_gated'],np.where(a['dwell_valid'],a['dwell'],0)))
            np.testing.assert_array_equal(b['combined'],np.maximum(a['visual'],b['process']))
            for k in ['process','combined']:
                assert np.all(b[k]<=a[k]);np.testing.assert_array_equal(b[k][state<4],a[k][state<4])


def prepare():
    for e,b in PAIRS.items():
        cfg=config(b);cfg.update(experiment=e,scope=f'R04_same_selected_track_pair_transition_{e.split("_")[-1]}',transition_evidence_gate='same_track_pair');write(f'configs/experiment{e}.json',cfg)
    for e in ['31',*AFTER]:
        root=Path(f'artifacts/experiment{e}');root.mkdir(parents=True,exist_ok=True);Path(f'results/experiment{e}').mkdir(parents=True,exist_ok=True);link=root/'features'
        if not link.exists():link.symlink_to('../experiment30/reset/features',target_is_directory=True)
        assert (link/'R04').resolve()==ROOT.resolve()
    files=[*[Path(f'configs/experiment{e}.json') for e in VARIANTS],*[Path(f'artifacts/experiment{e}/normal_model.npz') for e in BEFORE],Path('results/stage00/splits.json'),Path('results/experiment18/process_discovery.json'),Path('docs/EXPERIMENT31_PLAN.md'),Path('tests/test_pair_transition_gate.py'),*sorted(Path('src/ipad_vad').glob('*.py')),*[Path('scripts')/s for s in ['experiment31_pair_gate.py','validate_pair_gate.py','experiment30_track_reset.py','evaluate_route_holdout.py','evaluate_baseline.py','audit_bank_dispatch.py','audit_request_calibration.py','audit_missing_age.py']],*[ROOT/f'training_{s}.npz' for s in SPLIT['fit']+SPLIT['calibration']]]
    freeze('pre_normal_protocol.json',files);print('Normal protocol frozen',flush=True)


def normal():
    verify_freeze('pre_normal_protocol.json');fit=[load_cache(ROOT,s) for s in SPLIT['fit']];cal=[load_cache(ROOT,s) for s in SPLIT['calibration']];models=fitted_models(fit);rows={};dest=ART/'full_normal';dest.mkdir(parents=True,exist_ok=True)
    for e,m in models.items():
        m.calibrate(cal);verify_references(m,cal);source=load(f'artifacts/experiment{PAIRS.get(e,e)}/normal_model.npz');array_checks(model_arrays(m),source,e in AFTER)
        r=[m.score(d) for d in cal];scores={k:np.concatenate([v[k] for v in r]) for k in KEYS};np.savez_compressed(dest/f'{e}_model.npz',**model_arrays(m));np.savez_compressed(dest/f'{e}_scores.npz',**scores);rows[e]=feasibility(m,cal)
        if e in BEFORE:array_checks(scores,load(f'artifacts/experiment{e}/normal_calibration_scores.npz'))
        print(e,rows[e],flush=True)
    paired_checks(models,fit+cal)
    cov=[coverage(d,s,part) for part in ['fit','calibration'] for s,d in zip(SPLIT[part],fit if part=='fit' else cal)]
    write(OUT/'normal_audit.json',{'normal_only':True,'variants':rows,'unchanged_pca_appearance_process_references_transition_dwell':True,'old_normal_models_scores_reproduced':True,'pair_gate_subset_of_observed_gate':True,'coverage':cov})


def pre_test():
    verify_freeze('pre_normal_protocol.json');audit=json.loads((OUT/'normal_audit.json').read_text());hold=json.loads((OUT/'normal_holdout.json').read_text());eligible=[]
    for e in AFTER:
        cells=[r for r in hold['folds'] if r['variant']==e]
        if audit['variants'][e]['eligible'] and len(cells)==5 and all(r['calibration_finite'] and r['calibration_unit_interval'] and r['normal_q99']<1 for r in cells):eligible.append(e)
    write(OUT/'test_eligibility.json',{'eligible':eligible,'failed':[e for e in AFTER if e not in eligible],'rule':'Full normal and all five calibration-video holdout folds finite, bounded and q99<1; fixed FIT support.'})
    freeze('pre_test_checkpoint.json',[OUT/'pre_normal_protocol.json',OUT/'normal_audit.json',OUT/'normal_holdout.json',OUT/'test_eligibility.json',*sorted((ART/'full_normal').glob('*.npz')),*sorted((ART/'normal_holdout').glob('*.npz')),*sorted(ROOT.glob('testing_*.npz'))]);print('Eligible:',eligible,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','normal','pre_test']);a=p.parse_args()
    with threadpool_limits(limits=4):globals()[a.stage]()
