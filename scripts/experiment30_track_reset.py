"""Staged normal-only preparation and audit of identity-bounded relation smoothing."""
import argparse,copy,json
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.scoring import Baseline
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from ipad_vad.data import hold_scores
from audit_transition_provenance import restore_model
from audit_missing_age import load,sha
from audit_bank_dispatch import verify_references
from evaluate_route_holdout import load_cache,normal_model_arrays

OUT=Path('results/experiment30');ART=Path('artifacts/experiment30')
SOURCE=Path('artifacts/experiment19/features/R04')
GROUPS=['control','reset'];GATES=['hold','pool','age']
VARIANTS=[f'30_{g}_{a}' for g in GROUPS for a in GATES]
KEYS=['visual','transition','dwell','process','combined','dwell_valid','dwell_age','dwell_reason','dwell_entry_context','transition_raw','transition_valid','transition_gated']
SPLIT=json.loads(Path('results/stage00/splits.json').read_text())['R04']


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def freeze(name,files):
    path=OUT/name;record={'created_at_utc':datetime.now(timezone.utc).isoformat(),'file_sha256':{str(p):sha(p) for p in files}}
    if path.exists():verify_freeze(name)
    else:write(path,record)


def verify_freeze(name):
    record=json.loads((OUT/name).read_text())
    for p,h in record['file_sha256'].items():assert sha(p)==h,p


def cfg(e):return json.loads(Path(f'configs/experiment{e}.json').read_text())

def root(g):return ART/g/'features/R04'

def arrays(model):
    a=normal_model_arrays(model);a['transition']=model.transition
    for (r,p),s in model.spaces.items():a[f'mean_{r}_{p}']=s.mean;a[f'basis_{r}_{p}']=s.basis
    for (a0,b),v in model.dwell.log_parameters.items():a[f'dwell_log_parameters_{a0}_{b}']=np.array(v)
    for (a0,b),v in model.dwell.context_durations.items():a[f'dwell_context_durations_{a0}_{b}']=v
    return a


def relation_models():
    old=restore_model({'relation_config':'configs/experiment18.json','relation_model':'artifacts/experiment18/normal_relation_model.npz'})
    new=copy.deepcopy(old);new.reset_on_track_change=True
    return {'control':old,'reset':new}


def transform(part,sequences):
    models=relation_models();raw_model=copy.deepcopy(models['control']);raw_model.smoothing=1;rows=[]
    for seq in sequences:
        source=SOURCE/f'{part}_{seq}.npz';d=load(source);raw,rv,rc=raw_model.descriptors(d);results={};details={}
        for group,m in models.items():
            p,v,c,x=m.transform(d);np.testing.assert_array_equal(v,rv);np.testing.assert_array_equal(c,rc)
            # Independent reconstruction from selected raw descriptors and track history.
            history=[];pair_history=[];last=None;expected_phase=0;mixed=boundaries=0
            for i in range(len(v)):
                if v[i]:
                    pair=tuple(d['tracks'][c[i]])
                    changed=last is not None and pair!=last;boundaries+=int(changed)
                    if group=='reset' and pair!=last:history=[];pair_history=[]
                    history=(history+[raw[i].astype(d['boxes'].dtype)])[-m.smoothing:];pair_history=(pair_history+[pair])[-m.smoothing:];last=pair
                    np.testing.assert_array_equal(x[i],np.mean(history,axis=0));mixed+=int(len(set(pair_history))>1)
                    expected_phase=int(np.argmin(np.sum(((x[i]-m.location)/m.scale-m.centers)**2,axis=1)))
                else:history=[];pair_history=[];last=None;assert not x[i].any()
                assert p[i]==expected_phase
            if group=='control':
                for key,value in [('phases',p),('relation_valid',v),('relation_detection_indices',c),('relation_descriptors',x)]:np.testing.assert_array_equal(d[key],value)
            else:assert mixed==0
            derived=dict(d,phases=p,relation_valid=v,relation_detection_indices=c,relation_descriptors=x)
            for k in d:
                if k not in ['phases','relation_descriptors']:np.testing.assert_array_equal(d[k],derived[k])
            dest=root(group)/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                saved=load(dest)
                for k in derived:np.testing.assert_array_equal(saved[k],derived[k])
            else:np.savez_compressed(dest,**derived)
            observed=v[1:]&v[:-1];counts=np.zeros((4,4),int);np.add.at(counts,(p[:-1][observed],p[1:][observed]),1)
            results[group]=derived;details[group]={'mixed_windows':mixed,'valid_samples':int(v.sum()),'track_boundaries':boundaries,'phase_counts':np.bincount(p,minlength=4).tolist(),'observed_phase_counts':np.bincount(p[v],minlength=4).tolist(),'observed_transition_counts':counts.tolist()}
        old,new=results['control'],results['reset'];changed=old['phases']!=new['phases'];dense=hold_scores(d['indices'],changed,int(d['frame_count']))
        rows.append({'sequence':seq,'part':part,'partition':('fit' if seq in SPLIT['fit'] else 'calibration') if part=='training' else 'test','samples':len(changed),'frames':int(d['frame_count']),'phase_changed_samples':int(changed.sum()),'phase_changed_frames':int(dense.sum()),'descriptor_changed_samples':int(np.any(old['relation_descriptors']!=new['relation_descriptors'],axis=1).sum()),'groups':details})
    return rows


def prepare():
    files=[Path('docs/EXPERIMENT30_PLAN.md'),Path('scripts/experiment30_track_reset.py'),Path('scripts/diagnose_track_reset.py'),Path('scripts/evaluate_baseline.py'),Path('scripts/audit_bank_dispatch.py'),Path('scripts/audit_request_calibration.py'),Path('scripts/evaluate_route_holdout.py'),Path('scripts/audit_transition_provenance.py'),Path('scripts/audit_missing_age.py'),Path('tests/test_track_reset.py'),Path('configs/experiment18.json'),Path('artifacts/experiment18/normal_relation_model.npz'),Path('artifacts/experiment17/anchor_text_features.npz'),Path('results/experiment18/process_discovery.json'),Path('results/stage00/splits.json'),*[Path(f'configs/experiment27_{g}.json') for g in GATES],*sorted(Path('src/ipad_vad').glob('*.py')),*[SOURCE/f'training_{s}.npz' for s in SPLIT['fit']+SPLIT['calibration']]]
    freeze('pre_normal_protocol.json',files)
    for g in GROUPS:
        root(g).mkdir(parents=True,exist_ok=True)
        for gate in GATES:
            link=Path(f'artifacts/experiment30_{g}_{gate}/features');link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to(f'../experiment30/{g}/features',target_is_directory=True)
    rows=transform('training',SPLIT['fit']+SPLIT['calibration']);write(OUT/'normal_transform.json',{'normal_only':True,'rows':rows,'unchanged_selection_valid_raw_features_and_frozen_centers':True})
    natural={};models={}
    for g in GROUPS:
        base=cfg('27_hold');base.pop('appearance_phase_ranks');m=Baseline(base,load_process(base));m.fit([load_cache(root(g),s) for s in SPLIT['fit']]);models[g]=m
        natural[g]={f'{r}:{p}':{'samples':s.n,'rank':s.rank} for (r,p),s in sorted(m.spaces.items())}
    limits=cfg('27_hold')['appearance_phase_ranks'];maps={g:{} for g in GROUPS}
    for g in GROUPS:
        for key,value in natural[g].items():
            if key.endswith(':-1'):continue
            other=natural['reset' if g=='control' else 'control'].get(key)
            maps[g][key]=min(value['rank'],other['rank'],limits.get(key,32)) if other else value['rank']
    for key,s in models['control'].spaces.items():
        if key[1]==-1:
            other=models['reset'].spaces[key];assert s.n==other.n
            for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(s,attr),getattr(other,attr))
    write(OUT/'fit_rank_control.json',{'normal_fit_only':True,'natural_banks':natural,'rank_maps':maps,'exp27_rank_limits':limits,'shared_phase_banks':sorted(set(maps['control'])&set(maps['reset'])),'only_control':sorted(set(maps['control'])-set(maps['reset'])),'only_reset':sorted(set(maps['reset'])-set(maps['control'])),'pooled_banks_unchanged':True,'control_rank_map_matches_27':maps['control']==limits})
    for group in GROUPS:
        for gate in GATES:
            e=f'30_{group}_{gate}';options=cfg(f'27_{gate}');options.update(experiment=e,scope=f'R04_track_boundary_{group}_{gate}',feature_source_experiment=f'30/{group}',appearance_phase_ranks=maps[group]);options['relational_phase']['reset_on_track_change']=group=='reset';options['relation_model_source']='artifacts/experiment18/normal_relation_model.npz';write(f'configs/experiment{e}.json',options)
    freeze('pre_calibration_checkpoint.json',[OUT/'pre_normal_protocol.json',OUT/'normal_transform.json',OUT/'fit_rank_control.json',*[Path(f'configs/experiment{e}.json') for e in VARIANTS],*[p for g in GROUPS for p in sorted(root(g).glob('training_*.npz'))]])
    print(json.dumps({'rank_maps':maps,'normal_rows':len(rows)},indent=2),flush=True)


def feasibility(m,caches):
    rs=[m.score(d) for d in caches];finite=all(np.isfinite(r[k]).all() for r in rs for k in ['visual','transition','dwell','process','combined']);bounds=all(((r[k]>=0)&(r[k]<=1)).all() for r in rs for k in ['visual','transition','dwell','process','combined'])
    return {'finite':bool(finite),'unit_interval':bool(bounds),'q99':m.threshold,'eligible':bool(finite and bounds and m.threshold<1),'calibration_alarms':int(sum(np.sum(r['combined']>m.threshold) for r in rs)),'calibration_at_one':int(sum(np.sum(r['combined']==1) for r in rs))}


def normal():
    verify_freeze('pre_normal_protocol.json');verify_freeze('pre_calibration_checkpoint.json');full={};folds=[];eligible=[]
    for e in VARIANTS:
        g=e.split('_')[1];options=cfg(e);fit=[load_cache(root(g),s) for s in SPLIT['fit']];cal={s:load_cache(root(g),s) for s in SPLIT['calibration']}
        try:
            base=LognormalDwellBaseline(options,load_process(options));base.fit(fit)
            if e.endswith('age'):assert base.missing_age.tau==56
        except ValueError as error:
            full[e]={'eligible':False,'fit_failure':str(error)};continue
        model=copy.deepcopy(base);model.calibrate(list(cal.values()));verify_references(model,list(cal.values()));row=feasibility(model,list(cal.values()));row.update(context_support=model.dwell.context_support,supported_contexts=[list(k) for k in model.dwell.context_durations],log_parameters={str(k):list(v) for k,v in model.dwell.log_parameters.items()},process_reference_samples=len(model.process_reference),state_reference_samples={str(k):len(v) for k,v in model.state_process_references.items()});full[e]=row
        dest=ART/'full_normal';dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/f'{e}_model.npz',**arrays(model));np.savez_compressed(dest/f'{e}_scores.npz',**{k:np.concatenate([model.score(d)[k] for d in cal.values()]) for k in KEYS})
        for held,d in cal.items():
            used=[s for s in cal if s!=held];m=copy.deepcopy(base);used_caches=[cal[s] for s in used];m.calibrate(used_caches);verify_references(m,used_caches);r=m.score(d);v=feasibility(m,used_caches);alarm=hold_scores(d['indices'],r['combined']>m.threshold,int(d['frame_count'])).astype(bool)
            v.update(variant=e,held_out=held,calibration_sequences=used,held_out_frames=len(alarm),held_out_frame_alarms=int(alarm.sum()),held_out_sample_alarms=int(np.sum(r['combined']>m.threshold)),held_out_finite=all(bool(np.isfinite(r[k]).all()) for k in KEYS),held_out_bounded=all(bool(((r[k]>=0)&(r[k]<=1)).all()) for k in ['visual','transition','dwell','process','combined']))
            v['eligible']=v['eligible'] and v['held_out_finite'] and v['held_out_bounded'];folds.append(v);dest=ART/'normal_holdout';dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/f'{e}_exclude_{held}_model.npz',**arrays(m));np.savez_compressed(dest/f'{e}_exclude_{held}_scores.npz',**{k:r[k] for k in KEYS},indices=d['indices'],phases=d['phases'])
        if row['eligible'] and all(r['eligible'] for r in folds if r['variant']==e):eligible.append(e)
        print(e,row['q99'],'holdout FP',sum(r['held_out_frame_alarms'] for r in folds if r['variant']==e),flush=True)
    write(OUT/'normal_audit.json',{'normal_only':True,'variants':full,'folds':folds,'eligible':eligible,'failed':[e for e in VARIANTS if e not in eligible]})
    freeze('pre_test_checkpoint.json',[OUT/'pre_normal_protocol.json',OUT/'pre_calibration_checkpoint.json',OUT/'normal_audit.json',*sorted((ART/'full_normal').glob('*.npz')),*sorted((ART/'normal_holdout').glob('*.npz'))])


def test_prepare():
    for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json','pre_test_checkpoint.json']:verify_freeze(name)
    audit=json.loads((OUT/'normal_audit.json').read_text());assert audit['eligible']
    paths=sorted(SOURCE.glob('testing_*.npz'));freeze('test_input_checkpoint.json',paths)
    rows=transform('testing',[p.stem.split('_')[1] for p in paths]);write(OUT/'test_transform.json',{'rows':rows,'eligible_variants':audit['eligible'],'no_labels_used_for_transform':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','normal','test_prepare']);a=p.parse_args()
    with threadpool_limits(limits=4):globals()[a.stage]()
