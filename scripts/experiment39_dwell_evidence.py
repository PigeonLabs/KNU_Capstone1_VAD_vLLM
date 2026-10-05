"""Staged normal-only audit of same-pair evidence in dwell fusion."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.data import hold_scores
from ipad_vad.dwell_evidence import same_pair_since_entry_mask
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from experiment38_initial_observation import protected_inputs as previous_inputs
import experiment30_track_reset as common
from experiment30_track_reset import write,load,sha,arrays,load_cache

OUT=Path('results/experiment39');ART=Path('artifacts/experiment39');SOURCE=Path('artifacts/experiment37/asinh/features/R04')
GROUPS=['control','gated'];GATES=['hold','pool','age'];VARIANTS=[f'39_{g}_{a}' for g in GROUPS for a in GATES];SPLIT=common.SPLIT
BASE_KEYS=list(common.KEYS);KEYS=BASE_KEYS+['dwell_evidence_valid','dwell_gated']
RAW_KEYS=[k for k in BASE_KEYS if k not in ['process','combined']]
REASONS={'0':'continuous_observed_entry','1':'relation_missing','2':'entry_not_observed','3':'pair_changed_since_or_at_entry'}

def root(g):return SOURCE
def cfg(e):return json.loads(Path(f'configs/experiment{e}.json').read_text())
def configure_common():
    common.OUT=OUT;common.ART=ART;common.GROUPS=GROUPS;common.GATES=GATES;common.VARIANTS=VARIANTS;common.root=root;common.cfg=cfg;common.KEYS=KEYS

def protected_inputs():
    return previous_inputs()+[*[Path(f'artifacts/experiment38_guarded_{g}/{n}.npz') for g in GATES for n in ['normal_model','normal_calibration_scores']],*sorted(Path('artifacts/experiment38/full_normal').glob('*.npz')),*sorted(Path('artifacts/experiment38/normal_holdout').glob('*.npz'))]

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

def scalar_evidence(d):
    """Independent scalar reference; gather selected identities without edge helper."""
    n=len(d['phases']);evidence=np.zeros(n,bool);reason=np.full(n,2,int);known=False;cause=2;previous=None;previous_phase=None
    for i in range(n):
        if not d['relation_valid'][i]:
            known=False;cause=1;previous=None;reason[i]=1;previous_phase=None;continue
        pair=tuple(int(d['tracks'][j]) for j in d['relation_detection_indices'][i])
        if previous is None:known=False;cause=2
        elif pair!=previous:known=False;cause=3
        elif d['phases'][i]!=previous_phase:known=True;cause=0
        evidence[i]=known;reason[i]=0 if known else cause;previous=pair;previous_phase=d['phases'][i]
    return evidence,reason

def prefix(d,end):
    return {k:(d[k][d['object_frames']<end] if k in ['tracks','object_frames'] else d[k][:end]) for k in ['phases','relation_valid','relation_detection_indices','tracks','object_frames']}

def prepare():
    if (OUT/'pre_calibration_checkpoint.json').exists():
        for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json']:common.verify_freeze(name)
        print('Preparation already frozen',flush=True);return
    opened=normal_guard()
    for g in GROUPS:
        for gate in GATES:
            e=f'39_{g}_{gate}';options=cfg(f'38_guarded_{gate}');options.update(experiment=e,scope=f'R04_dwell_evidence_{g}_{gate}',dwell_evidence_gate='ungated' if g=='control' else 'same_track_pair_since_entry');write(f'configs/experiment{e}.json',options)
            link=Path(f'artifacts/experiment{e}/features');link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to('../experiment37/asinh/features',target_is_directory=True)
            assert (link/'R04').resolve()==SOURCE.resolve()
    files=[Path('docs/EXPERIMENT39_PLAN.md'),Path(__file__),Path('tests/test_dwell_evidence.py'),*sorted(Path('src/ipad_vad').glob('*.py')),*[Path('scripts')/n for n in ['experiment30_track_reset.py','evaluate_baseline.py','audit_bank_dispatch.py','evaluate_route_holdout.py','audit_missing_age.py']],Path('results/stage00/splits.json'),Path('results/experiment37/process_discovery.json'),*[Path(f'configs/experiment38_guarded_{g}.json') for g in GATES],*[Path(f'configs/experiment{e}.json') for e in VARIANTS],*protected_inputs()]
    common.freeze('pre_normal_protocol.json',files);rows=[];checks=0
    for part in ['fit','calibration']:
        for seq in SPLIT[part]:
            d=load(SOURCE/f'training_{seq}.npz');mask,reason=scalar_evidence(d);np.testing.assert_array_equal(same_pair_since_entry_mask(d),mask)
            for end in [len(mask)//2,len(mask)-1]:np.testing.assert_array_equal(same_pair_since_entry_mask(prefix(d,end)),mask[:end]);checks+=1
            rows.append({'partition':part,'sequence':seq,'samples':len(mask),'frames':int(d['frame_count']),'evidence_samples':int(mask.sum()),'evidence_frames':int(hold_scores(d['indices'],mask,int(d['frame_count'])).sum()),'reason_samples':{str(k):int(np.sum(reason==k)) for k in range(4)}})
    write(OUT/'normal_evidence.json',{'rows':rows,'prefix_checks':checks,'normal_only':True,'source_features_unchanged':True,'reason_codes':REASONS});common.freeze('pre_calibration_checkpoint.json',[OUT/'pre_normal_protocol.json',OUT/'normal_evidence.json']);write(OUT/'prepare_access.json',{'opened_normal_numeric_data':sorted(opened),'test_data_opened':False});print('Normal evidence protocol frozen:',len(rows),'videos',flush=True)

def compare_models(a,b):
    aa,bb=arrays(a),arrays(b);assert set(aa)==set(bb)
    for k in aa:
        if k!='threshold':np.testing.assert_array_equal(aa[k],bb[k],err_msg=k)
    assert a.request_calibration.support==b.request_calibration.support
    return {'control_q99':a.threshold,'gated_q99':b.threshold,'threshold_changed':a.threshold!=b.threshold}

def compare_scores(a,b,d):
    mask,reason=scalar_evidence(d);old,new=a.score(d),b.score(d)
    for key in RAW_KEYS:np.testing.assert_array_equal(old[key],new[key],err_msg=key)
    for key in ['dwell_evidence_valid']:np.testing.assert_array_equal(old[key],mask);np.testing.assert_array_equal(new[key],mask)
    np.testing.assert_array_equal(old['dwell_gated'],np.where(old['dwell_valid'],old['dwell'],0))
    np.testing.assert_array_equal(new['dwell_gated'],np.where(mask&old['dwell_valid'],old['dwell'],0))
    np.testing.assert_array_equal(new['process'],np.maximum(old['transition_gated'],new['dwell_gated']))
    for key in ['process','combined']:
        assert np.all(new[key]<=old[key]);np.testing.assert_array_equal(old[key][mask],new[key][mask])
    for x,y in zip(old['objects'],new['objects']):
        for u,v in zip(x,y):np.testing.assert_array_equal(u,v)
    n=int(d['frame_count']);dense=lambda x:hold_scores(d['indices'],x,n)
    used=old['dwell_valid'];after=used&mask;blocked=used&~mask
    return {'dwell_valid_samples':int(used.sum()),'dwell_eligible_samples':int(after.sum()),'dwell_valid_frames':int(dense(used).sum()),'dwell_eligible_frames':int(dense(after).sum()),'blocked_reason_frames':{str(k):int(dense(blocked&(reason==k)).sum()) for k in range(1,4)},'process_changed_frames':int(dense(new['process']!=old['process']).sum()),'combined_changed_frames':int(dense(new['combined']!=old['combined']).sum()),'control_fp':int(dense(old['combined']>a.threshold).sum()),'gated_fp':int(dense(new['combined']>b.threshold).sum()),'gated_fixed_q99_fp':int(dense(new['combined']>a.threshold).sum())}

def normal():
    if (OUT/'normal_verification_checkpoint.json').exists():
        for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json','pre_test_checkpoint.json','normal_verification_checkpoint.json']:common.verify_freeze(name)
        print('Normal results already frozen',flush=True);return
    opened=normal_guard();common.normal();fit=[load_cache(SOURCE,s) for s in SPLIT['fit']];cal={s:load_cache(SOURCE,s) for s in SPLIT['calibration']};rows=[];changes={};exact=0
    for gate in GATES:
        a=LognormalDwellBaseline(cfg(f'39_control_{gate}'),load_process(cfg(f'39_control_{gate}')));a.fit(fit);b=LognormalDwellBaseline(cfg(f'39_gated_{gate}'),load_process(cfg(f'39_gated_{gate}')));b.fit(fit)
        for held in [None,*SPLIT['calibration']]:
            aa,bb=copy.deepcopy(a),copy.deepcopy(b);used=[d for s,d in cal.items() if s!=held];aa.calibrate(used);bb.calibrate(used);changes[f'{gate}_{held or "full"}']=compare_models(aa,bb)
            if held is None:
                for g,m in [('control',aa),('gated',bb)]:
                    saved=load(ART/'full_normal'/f'39_{g}_{gate}_model.npz')
                    for k,v in arrays(m).items():np.testing.assert_array_equal(v,saved[k],err_msg=k)
                prior=load(f'artifacts/experiment38_guarded_{gate}/normal_model.npz')
                for k,v in arrays(aa).items():np.testing.assert_array_equal(v,prior[k],err_msg=k)
                prior=load(f'artifacts/experiment38_guarded_{gate}/normal_calibration_scores.npz');score=load(ART/'full_normal'/f'39_control_{gate}_scores.npz')
                for k in BASE_KEYS:np.testing.assert_array_equal(score[k],prior[k],err_msg=k)
                exact+=1;probes=[('fit',s,d) for s,d in zip(SPLIT['fit'],fit)]+[('calibration',s,d) for s,d in cal.items()]
            else:probes=[('holdout',held,cal[held])]
            for part,seq,d in probes:rows.append(dict(gate=gate,partition=part,sequence=seq,held_out=held,**compare_scores(aa,bb,d)))
    write(OUT/'normal_fusion_audit.json',{'rows':rows,'threshold_comparisons':changes,'control_normal_models_and_prior_score_keys_exact_to38':exact,'all_non_threshold_model_arrays_unchanged':True,'raw_scores_unchanged':True,'gated_process_and_combined_nonincreasing':True});write(OUT/'normal_access.json',{'opened_normal_numeric_data':sorted(opened),'test_data_opened':False});common.freeze('normal_verification_checkpoint.json',[OUT/'pre_test_checkpoint.json',OUT/'normal_fusion_audit.json',OUT/'normal_access.json']);print('Normal fusion checks passed',flush=True)

def test_prepare():
    for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json','pre_test_checkpoint.json','normal_verification_checkpoint.json']:common.verify_freeze(name)
    audit=json.loads((OUT/'normal_audit.json').read_text());assert audit['eligible'];paths=sorted(SOURCE.glob('testing_*.npz'));common.freeze('test_input_checkpoint.json',paths)
    write(OUT/'test_input_manifest.json',{'paths':[str(p) for p in paths],'features_reused_exactly':True,'no_new_phase_fit':True,'eligible':audit['eligible']});print('Frozen test inputs:',len(paths),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','normal','test_prepare']);a=p.parse_args();configure_common()
    with threadpool_limits(limits=4):globals()[a.stage]()
