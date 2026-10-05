"""Freeze and evaluate appearance routing before the first observed relation."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.data import hold_scores
from ipad_vad.scoring import observations
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from experiment37_asinh_phase import protected_inputs as previous_inputs
import experiment30_track_reset as common
from experiment30_track_reset import write,load,sha,arrays,load_cache,KEYS

OUT=Path('results/experiment38');ART=Path('artifacts/experiment38');SOURCE=Path('artifacts/experiment37/asinh/features/R04');GROUPS=['control','guarded'];GATES=['hold','pool','age'];VARIANTS=[f'38_{g}_{a}' for g in GROUPS for a in GATES];SPLIT=common.SPLIT
PROCESS_KEYS=[k for k in KEYS if k not in ['visual','combined']]

def root(g):return SOURCE
def cfg(e):return json.loads(Path(f'configs/experiment{e}.json').read_text())
def configure_common():
    common.OUT=OUT;common.ART=ART;common.GROUPS=GROUPS;common.GATES=GATES;common.VARIANTS=VARIANTS;common.root=root;common.cfg=cfg

def protected_inputs():
    return previous_inputs()+[Path('artifacts/experiment37/normal_relation_model.npz'),*[SOURCE/f'training_{s}.npz' for s in SPLIT['fit']+SPLIT['calibration']],*[Path(f'artifacts/experiment37_asinh_{g}/{n}.npz') for g in GATES for n in ['normal_model','normal_calibration_scores']],*sorted(Path('artifacts/experiment37/full_normal').glob('*.npz')),*sorted(Path('artifacts/experiment37/normal_holdout').glob('*.npz'))]

def normal_guard():
    allowed={str(p.resolve()) for p in protected_inputs()+[Path('artifacts/experiment18/normal_relation_model.npz'),Path('artifacts/experiment17/anchor_text_features.npz')]};opened=set()
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]);s=str(p.resolve())
        if '/test_label/' in s or '/predictions/' in s or p.name.startswith('testing_') or ('/testing/' in s and p.suffix not in ['.py','.pyc']):raise RuntimeError('Test access forbidden before normal checkpoint: '+s)
        if p.suffix.lower() in ['.npz','.npy','.jpg','.png','.mp4']:
            if s not in allowed and not p.resolve().is_relative_to(ART.resolve()):raise RuntimeError('Outside normal numeric allowlist: '+s)
            opened.add(s)
    sys.addaudithook(hook);return opened

def scalar_seen(data):
    seen=[];observed=False
    for valid in data['relation_valid']:
        observed=observed or bool(valid);seen.append(observed)
    return np.array(seen,bool)

def prepare():
    if (OUT/'pre_calibration_checkpoint.json').exists():
        for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json']:common.verify_freeze(name)
        print('Preparation already frozen',flush=True);return
    opened=normal_guard()
    for g in GROUPS:
        for gate in GATES:
            e=f'38_{g}_{gate}';options=cfg(f'37_asinh_{gate}');options.update(experiment=e,scope=f'R04_initial_observation_{g}_{gate}',feature_source_experiment='37/asinh',appearance_initial_observation_gate=g=='guarded');write(f'configs/experiment{e}.json',options)
            link=Path(f'artifacts/experiment{e}/features');link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to('../experiment37/asinh/features',target_is_directory=True)
            assert (link/'R04').resolve()==SOURCE.resolve()
    files=[Path('docs/EXPERIMENT38_PLAN.md'),Path(__file__),Path('tests/test_initial_observation.py'),*sorted(Path('src/ipad_vad').glob('*.py')),*[Path('scripts')/n for n in ['experiment30_track_reset.py','evaluate_baseline.py','audit_bank_dispatch.py','evaluate_route_holdout.py','audit_missing_age.py']],Path('results/stage00/splits.json'),Path('results/experiment37/process_discovery.json'),*[Path(f'configs/experiment37_asinh_{g}.json') for g in GATES],*[Path(f'configs/experiment{e}.json') for e in VARIANTS],*protected_inputs()]
    common.freeze('pre_normal_protocol.json',files);rows=[]
    for part in ['fit','calibration']:
        for seq in SPLIT[part]:
            d=load(SOURCE/f'training_{seq}.npz');seen=scalar_seen(d);dense=hold_scores(d['indices'],~seen,int(d['frame_count']));where=np.flatnonzero(d['relation_valid'])
            rows.append({'partition':part,'sequence':seq,'samples':len(seen),'frames':int(d['frame_count']),'initial_unobserved_samples':int((~seen).sum()),'initial_unobserved_frames':int(dense.sum()),'first_observed_source_frame':int(d['indices'][where[0]]) if len(where) else None,'valid_samples':int(d['relation_valid'].sum()),'original_initial_phase':int(d['phases'][0])})
    write(OUT/'normal_initial_observations.json',{'rows':rows,'normal_only':True,'source_features_unchanged':True,'new_phase_fit':False});common.freeze('pre_calibration_checkpoint.json',[OUT/'pre_normal_protocol.json',OUT/'normal_initial_observations.json']);write(OUT/'prepare_access.json',{'opened_normal_numeric_data':sorted(opened),'test_data_opened':False});print('Normal protocol frozen:',len(rows),'videos',flush=True)

def compare_models(a,b):
    aa,bb=arrays(a),arrays(b);assert set(aa)==set(bb);changed=[]
    for k in aa:
        if k=='threshold' or k.startswith('calibration_'):
            if not np.array_equal(aa[k],bb[k]):changed.append(k)
        else:np.testing.assert_array_equal(aa[k],bb[k],err_msg=k)
    assert a.request_calibration.support==b.request_calibration.support
    return changed

def compare_scores(a,b,d,gate):
    seen=scalar_seen(d);old=a.score(d);new=b.score(d);requested=a.appearance_phases(d);expected=requested.copy();expected[~seen]=-1;np.testing.assert_array_equal(b.appearance_phases(d),expected)
    for key in PROCESS_KEYS:np.testing.assert_array_equal(old[key],new[key],err_msg=key)
    for key in ['visual','combined']:np.testing.assert_array_equal(old[key][seen],new[key][seen],err_msg=key)
    actual=[]
    for (r,frames,x),(s,idx,y) in zip(a.raw(d)[0],b.raw(d)[0]):
        assert r==s;np.testing.assert_array_equal(frames,idx);np.testing.assert_array_equal(x[seen[frames]],y[seen[frames]])
        for i in range(len(frames)):
            expected_key=b.appearance_space_key(r,int(expected[frames[i]]));assert int(b.appearance_routes(d,r,frames)[i])==int(expected_key[1]>=0)
        actual.append({'role':r,'initial_observations':int((~seen[frames]).sum()),'initial_phase_requests_before':int(np.sum((requested[frames]>=0)&~seen[frames])),'initial_phase_requests_after':int(np.sum((b.appearance_phases(d)[frames]>=0)&~seen[frames]))})
    for (_,frames,x),(_,_,y) in zip(old['objects'],new['objects']):np.testing.assert_array_equal(x[seen[frames]],y[seen[frames]])
    if gate!='hold':
        for key in KEYS:np.testing.assert_array_equal(old[key],new[key])
        for (_,_,x),(_,_,y) in zip(old['objects'],new['objects']):np.testing.assert_array_equal(x,y)
    dense=hold_scores(d['indices'],~seen,int(d['frame_count'])).astype(bool)
    return {'initial_samples':int((~seen).sum()),'initial_frames':int(dense.sum()),'visual_changed_samples':int(np.sum(old['visual']!=new['visual'])),'combined_changed_samples':int(np.sum(old['combined']!=new['combined'])),'initial_fp_control':int(np.sum(hold_scores(d['indices'],old['combined']>a.threshold,int(d['frame_count']))&dense)),'initial_fp_guarded':int(np.sum(hold_scores(d['indices'],new['combined']>b.threshold,int(d['frame_count']))&dense)),'after_first_fp_control':int(np.sum(hold_scores(d['indices'],old['combined']>a.threshold,int(d['frame_count']))&~dense)),'after_first_fp_guarded':int(np.sum(hold_scores(d['indices'],new['combined']>b.threshold,int(d['frame_count']))&~dense)),'requests_by_role':actual}

def normal():
    if (OUT/'normal_verification_checkpoint.json').exists():
        for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json','pre_test_checkpoint.json','normal_verification_checkpoint.json']:common.verify_freeze(name)
        print('Normal results already frozen',flush=True);return
    opened=normal_guard();common.normal();fit=[load_cache(SOURCE,s) for s in SPLIT['fit']];cal={s:load_cache(SOURCE,s) for s in SPLIT['calibration']};rows=[];changes={};exact=0
    for gate in GATES:
        a=LognormalDwellBaseline(cfg(f'38_control_{gate}'),load_process(cfg(f'38_control_{gate}')));a.fit(fit);b=LognormalDwellBaseline(cfg(f'38_guarded_{gate}'),load_process(cfg(f'38_guarded_{gate}')));b.fit(fit)
        for held in [None,*SPLIT['calibration']]:
            aa,bb=copy.deepcopy(a),copy.deepcopy(b);used=[d for s,d in cal.items() if s!=held];aa.calibrate(used);bb.calibrate(used);changes[f'{gate}_{held or "full"}']=compare_models(aa,bb)
            if held is None:
                for g,m in [('control',aa),('guarded',bb)]:
                    saved=load(ART/'full_normal'/f'38_{g}_{gate}_model.npz')
                    for k,v in arrays(m).items():np.testing.assert_array_equal(v,saved[k],err_msg=k)
                prior=load(f'artifacts/experiment37_asinh_{gate}/normal_model.npz')
                for k,v in arrays(aa).items():np.testing.assert_array_equal(v,prior[k],err_msg=k)
                prior=load(f'artifacts/experiment37_asinh_{gate}/normal_calibration_scores.npz');score=load(ART/'full_normal'/f'38_control_{gate}_scores.npz')
                for k,v in score.items():np.testing.assert_array_equal(v,prior[k],err_msg=k)
                exact+=1;probes=[('fit',s,d) for s,d in zip(SPLIT['fit'],fit)]+[('calibration',s,d) for s,d in cal.items()]
            else:probes=[('holdout',held,cal[held])]
            for part,seq,d in probes:rows.append(dict(gate=gate,partition=part,sequence=seq,held_out=held,**compare_scores(aa,bb,d,gate)))
    write(OUT/'normal_routing_audit.json',{'rows':rows,'allowed_array_changes':changes,'control_normal_models_and_scores_exact_to37':exact,'used_request_cdf_pca_process_unchanged':True,'post_first_scores_unchanged':True,'pool_age_scores_unchanged':True});write(OUT/'normal_access.json',{'opened_normal_numeric_data':sorted(opened),'test_data_opened':False});common.freeze('normal_verification_checkpoint.json',[OUT/'pre_test_checkpoint.json',OUT/'normal_routing_audit.json',OUT/'normal_access.json']);print('Normal routing checks passed',flush=True)

def test_prepare():
    for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json','pre_test_checkpoint.json','normal_verification_checkpoint.json']:common.verify_freeze(name)
    audit=json.loads((OUT/'normal_audit.json').read_text());assert audit['eligible'];paths=sorted(SOURCE.glob('testing_*.npz'));common.freeze('test_input_checkpoint.json',paths)
    write(OUT/'test_input_manifest.json',{'paths':[str(p) for p in paths],'features_reused_exactly':True,'no_new_phase_fit':True,'eligible':audit['eligible']});print('Frozen test inputs:',len(paths),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','normal','test_prepare']);a=p.parse_args();configure_common()
    with threadpool_limits(limits=4):globals()[a.stage]()
