"""Staged normal-only phase refitting under a frozen confirmed selection policy."""
import argparse,copy,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
import numpy.testing
from threadpoolctl import threadpool_limits
from ipad_vad.phase_refit import refit_phase_centers
from scipy.optimize import linear_sum_assignment
from experiment35_confirmed_anchor import models as previous_models, independently_check as check_selection, protected_inputs as previous_inputs
from ipad_vad.confirmed_anchor import ConfirmedAnchorPhase
from ipad_vad.scoring import Baseline
from ipad_vad.experiment import load_process
from ipad_vad.data import hold_scores
from ipad_vad.dwell_episodes import extract_dwell_episodes
from audit_transition_provenance import restore_model
from experiment32_duration_evidence import protected_paths
import experiment30_track_reset as common
from experiment30_track_reset import write,load,sha

OUT=Path('results/experiment36');ART=Path('artifacts/experiment36');SOURCE=Path('artifacts/experiment35/confirmed/features/R04');GROUPS=['control','refit'];GATES=['hold','pool','age'];VARIANTS=[f'36_{g}_{a}' for g in GROUPS for a in GATES];SPLIT=common.SPLIT

def root(g):return ART/g/'features/R04'
def protected_inputs():
    return previous_inputs()+[*[SOURCE/f'training_{s}.npz' for s in SPLIT['fit']+SPLIT['calibration']],*[Path(f'artifacts/experiment35_confirmed_{gate}/{name}.npz') for gate in GATES for name in ['normal_model','normal_calibration_scores']],*sorted(Path('artifacts/experiment35/full_normal').glob('*.npz')),*sorted(Path('artifacts/experiment35/normal_holdout').glob('*.npz'))]

def cfg(e):return json.loads(Path(f'configs/experiment{e}.json').read_text())
def configure_common():
    common.OUT=OUT;common.ART=ART;common.GROUPS=GROUPS;common.GATES=GATES;common.VARIANTS=VARIANTS;common.root=root;common.cfg=cfg

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

def models():
    old=previous_models()['confirmed'];new=copy.deepcopy(old)
    saved=load(ART/'normal_relation_model.npz')
    for key in ['location','scale','centers']:setattr(new,key,saved[key])
    np.testing.assert_array_equal(saved['area_roles'],list(old.area_upper))
    np.testing.assert_array_equal(saved['area_upper'],list(old.area_upper.values()))
    return {'control':old,'refit':new}

def independently_check(d,m,group,p,v,c,x):
    return check_selection(d,m,'confirmed',p,v,c,x)

def fit_phase():
    fit=[load(SOURCE/f'training_{s}.npz') for s in SPLIT['fit']]
    m=previous_models()['confirmed'];old=copy.deepcopy(m);refit_phase_centers(m,fit)
    ART.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(ART/'normal_relation_model.npz',location=m.location,scale=m.scale,centers=m.centers,area_roles=list(m.area_upper),area_upper=list(m.area_upper.values()))
    contingency=np.zeros((4,4),int);sequence_counts={};distances={g:[] for g in GROUPS};descriptors=[]
    for seq,d in zip(SPLIT['fit'],fit):
        before=old.transform(d);after=m.transform(d)
        for a,b in zip(before[1:],after[1:]):np.testing.assert_array_equal(a,b)
        p,v,c,x=before;q=after[0];np.add.at(contingency,(p[v],q[v]),1)
        sequence_counts[seq]={g:np.bincount(labels[v],minlength=4).tolist() for g,labels in [('control',p),('refit',q)]}
        descriptors.append(x[v])
        for g,model in [('control',old),('refit',m)]:
            distance=np.sum(((x[v,None,:]-model.location)/model.scale-model.centers)**2,axis=2).min(1);distances[g].extend(distance.tolist())
    a,b=linear_sum_assignment(-contingency);mapping={int(new):int(previous) for previous,new in zip(a,b)}
    x=np.concatenate(descriptors)
    evidence=dict(m.evidence,normal_fit_sequences=SPLIT['fit'],calibration_sequences_excluded=SPLIT['calibration'],fit_input_sha256={str(SOURCE/f'training_{s}.npz'):sha(SOURCE/f'training_{s}.npz') for s in SPLIT['fit']},raw_descriptor_std=x.std(0).tolist(),scaled_descriptor_std={g:((x-model.location)/model.scale).std(0).tolist() for g,model in [('control',old),('refit',m)]},nearest_center_squared_distance={g:{'mean':float(np.mean(v)),'median':float(np.median(v)),'max':float(np.max(v))} for g,v in distances.items()},distance_note='Different scaling: distances across models are not a common physical error metric.',normal_fit_contingency_control_rows_refit_columns=contingency.tolist(),diagnostic_refit_to_control_mapping=mapping,raw_fit_agreement=float(np.trace(contingency)/contingency.sum()),permutation_adjusted_fit_agreement=float(contingency[a,b].sum()/contingency.sum()),mapping_used_for_inference=False,per_video_observed_phase_counts=sequence_counts,old_location=old.location.tolist(),old_scale=old.scale.tolist(),old_centers=old.centers.tolist())
    write(OUT/'phase_fit.json',evidence)
    process=json.loads(Path('results/experiment18/process_discovery.json').read_text());process.update(derived_from='results/experiment18/process_discovery.json',relation_model=str(ART/'normal_relation_model.npz'),normal_phase_fit_sequences=SPLIT['fit'],note='Normal FIT phase centers refitted under experiment35 two-sample confirmed selection; frozen area gates. Same vocabulary and self/next/cycle template. No VLM call or action ground truth.')
    write(OUT/'process_discovery.json',process)

def transform(part,sequences):
    ms=models();rows=[];eps=[]
    for seq in sequences:
        d=load(SOURCE/f'{part}_{seq}.npz');results={};details={};partition=('fit' if seq in SPLIT['fit'] else 'calibration') if part=='training' else 'test'
        for g,m in ms.items():
            p,v,c,x=m.transform(d);decisions=independently_check(d,m,g,p,v,c,x);write(ART/'selection_traces'/g/f'{part}_{seq}.json',decisions)
            np.testing.assert_array_equal(m.margins(d),d['anchor_gate_margin']);np.testing.assert_array_equal(m.raw_margins(d),d['anchor_margin'])
            if g=='control':
                for key,val in [('phases',p),('relation_valid',v),('relation_detection_indices',c),('relation_descriptors',x)]:np.testing.assert_array_equal(d[key],val)
            derived=dict(d,phases=p,relation_valid=v,relation_detection_indices=c,relation_descriptors=x)
            for k in d:
                if k!='phases':np.testing.assert_array_equal(d[k],derived[k])
            np.testing.assert_array_equal(c,d['relation_detection_indices']);np.testing.assert_array_equal(x,d['relation_descriptors']);np.testing.assert_array_equal(v,d['relation_valid'])
            dest=root(g)/f'{part}_{seq}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                saved=load(dest)
                for k in derived:np.testing.assert_array_equal(derived[k],saved[k])
            else:np.savez_compressed(dest,**derived)
            episode=extract_dwell_episodes(derived)['episodes'];eps.extend(dict(e,group=g,partition=partition,sequence=seq) for e in episode)
            pair_changes=Counter()
            for i in range(1,len(v)):
                if v[i-1] and v[i]:
                    change=d['tracks'][c[i]]!=d['tracks'][c[i-1]]
                    if change.any():pair_changes['both' if change.all() else 'anchor' if change[0] else 'target']+=1
            validanchor=c[:,0]>=0
            details[g]={'valid_samples':int(v.sum()),'pair_changes':dict(pair_changes),'phase_counts':np.bincount(p,minlength=4).tolist(),'observed_phase_counts':np.bincount(p[v],minlength=4).tolist(),'episode_status':dict(Counter(e['status'] for e in episode)),'selection_decisions':dict(Counter(r['reason'] for r in decisions)),'selected_anchor_margin_mean':float(np.mean(d['anchor_gate_margin'][c[validanchor,0]])) if validanchor.any() else None}
            results[g]=derived
        a,b=results['control'],results['refit'];change=a['relation_detection_indices'][:,0]!=b['relation_detection_indices'][:,0];phase=a['phases']!=b['phases']
        rows.append({'sequence':seq,'partition':partition,'samples':len(v),'frames':int(d['frame_count']),'anchor_selection_changed_samples':int(change.sum()),'phase_changed_samples':int(phase.sum()),'phase_changed_frames':int(hold_scores(d['indices'],phase,int(d['frame_count'])).sum()),'descriptor_changed_samples':int(np.any(a['relation_descriptors']!=b['relation_descriptors'],axis=1).sum()),'groups':details})
    return rows,eps

def prepare():
    opened=normal_guard();files=[Path('docs/EXPERIMENT36_PLAN.md'),Path('scripts/experiment36_phase_refit.py'),Path('tests/test_phase_refit.py'),Path('scripts/experiment35_confirmed_anchor.py'),*sorted(Path('src/ipad_vad').glob('*.py')),*[Path('scripts')/n for n in ['experiment30_track_reset.py','evaluate_baseline.py','audit_transition_provenance.py','audit_bank_dispatch.py','evaluate_route_holdout.py','audit_missing_age.py']],*[Path(f'configs/experiment35_confirmed_{g}.json') for g in GATES],Path('configs/experiment34_semantic_hold.json'),Path('configs/experiment18.json'),Path('results/stage00/splits.json'),Path('results/experiment18/process_discovery.json'),Path('artifacts/experiment18/normal_relation_model.npz'),Path('artifacts/experiment17/anchor_text_features.npz'),*protected_inputs()]
    common.freeze('pre_normal_protocol.json',files)
    opened.clear();fit_phase();assert {p for p in opened if Path(p).name.startswith('training_')}=={str((SOURCE/f'training_{s}.npz').resolve()) for s in SPLIT['fit']};write(OUT/'phase_fit_access.json',{'opened_numeric_during_phase_fit':sorted(opened),'fit_sequences_only':SPLIT['fit'],'calibration_or_test_used':False})
    rows,episodes=transform('training',SPLIT['fit']+SPLIT['calibration']);write(OUT/'normal_transform.json',{'normal_only':True,'rows':rows,'valid_and_target_selection_unchanged':True,'independent_scalar_selection_descriptor_phase_checked':True});write(OUT/'normal_episodes.json',{'episodes':episodes})
    natural={};fitted={}
    for g in GROUPS:
        options=cfg('35_confirmed_hold');options.pop('appearance_phase_ranks');m=Baseline(options,load_process(options));m.fit([dict(load(root(g)/f'training_{s}.npz'),sequence_id=f'R04/training_{s}') for s in SPLIT['fit']]);fitted[g]=m
        natural[g]={f'{r}:{p}':{'samples':v.n,'rank':v.rank} for (r,p),v in m.spaces.items()}
    limits=cfg('35_confirmed_hold')['appearance_phase_ranks'];maps={g:{} for g in GROUPS}
    for g in GROUPS:
        other='refit' if g=='control' else 'control'
        for key,value in natural[g].items():
            if key.endswith(':-1'):continue
            peer=natural[other].get(key);maps[g][key]=min(value['rank'],peer['rank'] if peer else value['rank'],limits.get(key,32))
    for key,space in fitted['control'].spaces.items():
        if key[1]==-1:
            for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(space,attr),getattr(fitted['refit'].spaces[key],attr))
    write(OUT/'fit_rank_control.json',{'normal_fit_only':True,'natural_banks':natural,'rank_maps':maps,'previous_rank_limits':limits,'control_rank_map_matches_35':maps['control']==limits,'pooled_banks_unchanged':True,'shared_phase_banks':sorted(set(maps['control'])&set(maps['refit'])),'only_control':sorted(set(maps['control'])-set(maps['refit'])),'only_refit':sorted(set(maps['refit'])-set(maps['control']))})
    for g in GROUPS:
        for gate in GATES:
            e=f'36_{g}_{gate}';options=cfg(f'35_confirmed_{gate}');options.update(experiment=e,scope=f'R04_phase_refit_{g}_{gate}',feature_source_experiment=f'36/{g}',appearance_phase_ranks=maps[g],anchor_selection='confirmed_semantic_margin',anchor_confirmation_samples=2,relation_model_source=str(ART/'normal_relation_model.npz') if g=='refit' else 'artifacts/experiment18/normal_relation_model.npz',process_discovery=str(OUT/'process_discovery.json') if g=='refit' else 'results/experiment18/process_discovery.json');write(f'configs/experiment{e}.json',options)
            link=Path(f'artifacts/experiment{e}/features');link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to(f'../experiment36/{g}/features',target_is_directory=True)
    common.freeze('pre_calibration_checkpoint.json',[OUT/'pre_normal_protocol.json',OUT/'phase_fit.json',OUT/'phase_fit_access.json',OUT/'process_discovery.json',ART/'normal_relation_model.npz',OUT/'normal_transform.json',OUT/'normal_episodes.json',OUT/'fit_rank_control.json',*[Path(f'configs/experiment{e}.json') for e in VARIANTS],*[p for g in GROUPS for p in sorted(root(g).glob('training_*.npz'))]])
    write(OUT/'prepare_access.json',{'opened_normal_data':sorted(opened),'test_data_opened':False});print(json.dumps({'rank_maps':maps,'normal_sequences':len(rows)},indent=2),flush=True)

def normal():
    opened=normal_guard();common.normal()
    ranks=json.loads((OUT/'fit_rank_control.json').read_text());checks={}
    if ranks['control_rank_map_matches_35']:
        for gate in GATES:
            e=f'36_control_{gate}';saved=load(ART/'full_normal'/f'{e}_model.npz');old=load(f'artifacts/experiment35_confirmed_{gate}/normal_model.npz')
            for k,v in saved.items():np.testing.assert_array_equal(v,old[k],err_msg=k)
            scores=load(ART/'full_normal'/f'{e}_scores.npz');previous=load(f'artifacts/experiment35_confirmed_{gate}/normal_calibration_scores.npz')
            for k,v in scores.items():np.testing.assert_array_equal(v,previous[k],err_msg=k)
            checks[e]='model and calibration scores exactly reproduce experiment35_confirmed'
    else:checks['control']='rank changed; compare refitted matched-rank control, not identical to experiment35_confirmed'
    write(OUT/'control_reproduction.json',checks);write(OUT/'normal_access.json',{'opened_normal_data':sorted(opened),'test_data_opened':False})
    common.freeze('normal_verification_checkpoint.json',[OUT/'pre_test_checkpoint.json',OUT/'control_reproduction.json',OUT/'normal_access.json'])

def test_prepare():
    for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json','pre_test_checkpoint.json','normal_verification_checkpoint.json','pre_evaluation_review_checkpoint.json']:common.verify_freeze(name)
    audit=json.loads((OUT/'normal_audit.json').read_text());assert audit['eligible']
    paths=sorted(SOURCE.glob('testing_*.npz'));common.freeze('test_input_checkpoint.json',paths)
    rows,episodes=transform('testing',[p.stem.split('_')[1] for p in paths]);write(OUT/'test_transform.json',{'rows':rows,'no_labels_used_for_transform':True,'eligible_variants':audit['eligible']});write(OUT/'test_episode_diagnostic.json',{'episodes':episodes,'not_fit_or_calibration_data':True})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','normal','test_prepare']);args=parser.parse_args();configure_common()
    with threadpool_limits(limits=4):globals()[args.stage]()
