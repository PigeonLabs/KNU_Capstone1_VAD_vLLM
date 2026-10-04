"""Independently reconstruct normal route CDFs, holdout and test predictions."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from ipad_vad.data import hold_scores,evaluation_labels
from evaluate_route_holdout import load_cache,normal_model_arrays,support_rows


def load(p):
    with np.load(p,allow_pickle=False) as f:return dict(f)


def direct_routes(model,d,role,frames):
    return np.array([int(bool(d['relation_valid'][i]) and (role,int(d['phases'][i])) in model.spaces) for i in frames],dtype=np.int8)


def cdf(ref,x):
    ref=np.sort(ref)
    return (np.searchsorted(ref,x,side='left')+np.searchsorted(ref,x,side='right'))/(2*len(ref))


def independent_references(model,caches):
    pooled={};parts={};videos={}
    for d in caches:
        for role,frames,residual in model.raw(d)[0]:
            pooled.setdefault(role,[]).append(residual);routes=direct_routes(model,d,role,frames)
            np.testing.assert_array_equal(routes,model.appearance_routes(d,role,frames))
            for route in np.unique(routes):
                key=(role,int(route));parts.setdefault(key,[]).append(residual[routes==route]);videos.setdefault(key,set()).add(d['sequence_id'])
    pooled={k:np.concatenate(v) for k,v in pooled.items()};parts={k:np.concatenate(v) for k,v in parts.items()}
    for k,v in pooled.items():np.testing.assert_array_equal(v,model.calibration[k])
    refs={}
    if model.route_calibration is not None:
        for k,v in parts.items():
            assert model.route_calibration.support[k]['videos']==sorted(videos[k])
            if len(v)>=50 and len(videos[k])>=2:refs[k]=v
        assert set(refs)==set(model.route_calibration.references)
        for k,v in refs.items():np.testing.assert_array_equal(v,model.route_calibration.references[k])
    return pooled,refs


def independent_visual(model,d,pooled,refs):
    visual=np.zeros(len(d['indices']));objects=[]
    for role,frames,residual in model.raw(d)[0]:
        routes=direct_routes(model,d,role,frames);scores=cdf(pooled[role],residual)
        for route in np.unique(routes):
            key=(role,int(route));mask=routes==route
            if key in refs:scores[mask]=cdf(refs[key],residual[mask])
        np.maximum.at(visual,frames,scores);objects.append(scores)
    return visual,np.concatenate(objects)


def metric(y,s):return {'frames':len(y),'positive_frames':int(y.sum()),'auroc':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,'average_precision':float(average_precision_score(y,s)) if np.any(y==1) else None}


def main():
    out=Path('results/experiment20');root=Path('artifacts/experiment20');source=Path('artifacts/experiment19/features/R04');features=root/'features/R04';split=json.loads(Path('results/stage00/splits.json').read_text())['R04']
    protocol=json.loads((out/'pre_normal_protocol.json').read_text());checkpoint=json.loads((out/'pre_test_checkpoint.json').read_text())
    for record in [protocol,checkpoint]:
        for p,h in record['file_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    paths=sorted(source.glob('*.npz'));assert len(paths)==44
    for p in paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256((features/p.name).read_bytes()).hexdigest()==protocol['source_features_sha256'][p.name]
    fit=[load_cache(features,s) for s in split['fit']];cal={s:load_cache(features,s) for s in split['calibration']};configs={e:json.loads(Path(f'configs/experiment{e}.json').read_text()) for e in ['19','20']}
    holdout=json.loads((out/'normal_holdout.json').read_text());models={};process_keys=['transition','dwell','process','dwell_valid','dwell_age','dwell_reason','dwell_entry_context'];normal_q={};full_refs={}
    with threadpool_limits(limits=4):
        for e,cfg in configs.items():
            model=LognormalDwellBaseline(cfg,load_process(cfg));model.fit(fit);models[e]=model
        saved19=load('artifacts/experiment19/normal_model.npz');saved20=load(root/'normal_model.npz')
        for key,space in models['20'].spaces.items():
            for kind in ['mean','basis']:
                val=getattr(space,kind);np.testing.assert_array_equal(val,getattr(models['19'].spaces[key],kind));np.testing.assert_array_equal(val,saved19[f'{kind}_{key[0]}_{key[1]}']);np.testing.assert_array_equal(val,saved20[f'{kind}_{key[0]}_{key[1]}'])
        for row in holdout['folds']:
            e=row['variant'];held=row['held_out_sequence'];used=[s for s in split['calibration'] if s!=held];assert used==row['calibration_sequences']
            m=copy.deepcopy(models[e]);m.calibrate([cal[s] for s in used]);pooled,refs=independent_references(m,[cal[s] for s in used]);assert support_rows(m)==row['route_support']
            saved=load(root/'normal_holdout'/f'{e}_exclude_{held}_references.npz');expected=normal_model_arrays(m);assert set(saved)==set(expected)
            for k,v in expected.items():np.testing.assert_array_equal(v,saved[k])
            cal_combined=[]
            for s in used:
                r=m.score(cal[s]);visual,_=independent_visual(m,cal[s],pooled,refs);cal_combined.append(np.maximum(visual,r['process']))
            q=float(np.quantile(np.concatenate(cal_combined),.99,method='higher'));assert q==m.threshold==row['normal_q99']
            d=cal[held];r=m.score(d);visual,objects=independent_visual(m,d,pooled,refs);np.testing.assert_array_equal(visual,r['visual']);np.testing.assert_array_equal(np.maximum(visual,r['process']),r['combined'])
            pred=load(root/'normal_holdout'/f'{e}_exclude_{held}_scores.npz')
            for k in ['visual','combined',*process_keys]:np.testing.assert_array_equal(r[k],pred[k])
            np.testing.assert_array_equal(objects,pred['object_scores']);alarm=r['combined']>q;dense=hold_scores(d['indices'],alarm,int(d['frame_count'])).astype(bool)
            assert len(alarm)==row['held_out_samples'] and int(alarm.sum())==row['held_out_sample_alarms'];assert len(dense)==row['held_out_frames'] and int(dense.sum())==row['held_out_frame_alarms']
            for key in ['False','True']:
                mask=hold_scores(d['indices'],d['relation_valid'],len(dense)).astype(bool)==(key=='True');assert row['relation_strata'][key]=={'frames':int(mask.sum()),'alarms':int(np.sum(mask&dense))}
            global_route=hold_scores(d['indices'],direct_routes(m,d,-1,np.arange(len(d['indices']))),len(dense))
            for kind,name in [(0,'pooled'),(1,'phase')]:
                mask=global_route==kind;assert row['global_bank_strata'][name]=={'frames':int(mask.sum()),'alarms':int(np.sum(mask&dense))}
            for item in row['object_bank_strata']:
                kind=int(item['route']=='phase');mask=(pred['object_roles']==item['role'])&(pred['object_routes']==kind)
                assert item['observations']==int(mask.sum()) and item['alarms']==int(np.sum(mask&(objects>q)))
                expected_conditional=int(mask.sum()) if (item['role'],kind) in refs else 0
                assert item['conditional_cdf_observations']==expected_conditional
        for e in ['19','20']:
            rows=[r for r in holdout['folds'] if r['variant']==e]
            for key in ['held_out_samples','held_out_sample_alarms','held_out_frames','held_out_frame_alarms','held_out_at_one_samples']:
                assert holdout['totals'][e][key]==sum(r[key] for r in rows)
        for e,m in models.items():
            m.calibrate(list(cal.values()));full_refs[e]=independent_references(m,list(cal.values()));normal_q[e]=m.threshold
        cal20=load(root/'normal_calibration_scores.npz');cal19=load('artifacts/experiment19/normal_calibration_scores.npz');pre=load(root/'preflight_calibration_scores.npz')
        for k,v in pre.items():np.testing.assert_array_equal(v,cal20[k])
        for k in process_keys:np.testing.assert_array_equal(cal19[k],cal20[k])
        for e,m in models.items():
            saved=saved19 if e=='19' else saved20
            for k,v in normal_model_arrays(m).items():np.testing.assert_array_equal(v,saved[k])
            saved_cal=cal19 if e=='19' else cal20
            for k in ['visual','combined',*process_keys]:np.testing.assert_array_equal(np.concatenate([m.score(d)[k] for d in cal.values()]),saved_cal[k])
        for p in paths:
            part,seq=p.stem.split('_');d=load_cache(features,seq,part);a=models['19'].raw(d);b=models['20'].raw(d)
            np.testing.assert_array_equal(a[1],b[1])
            for (_,fa,va),(_,fb,vb) in zip(a[0],b[0]):np.testing.assert_array_equal(fa,fb);np.testing.assert_array_equal(va,vb)
        y=[];valid=[];scores={e:{k:[] for k in ['visual','process','combined']} for e in ['19','20']};branch={k:{'normal':0,'anomaly':0} for k in ['visual','transition','dwell','process_only_over_visual']};usage={}
        preds=sorted((root/'predictions').glob('*.npz'));assert len(preds)==19
        for p in preds:
            seq=p.stem.split('_')[1];d=load_cache(features,seq,'testing');a=load(Path('artifacts/experiment19/predictions')/p.name);b=load(p);n=int(d['frame_count'])
            truth=evaluation_labels(np.load(Path('/media/jeong/ExtHDD/IPAD_vLLM/IPAD_dataset/R04/test_label')/f'{int(seq):03}.npy'),n);np.testing.assert_array_equal(truth,b['labels'])
            for k in ['labels','indices','phases','boxes','tracks','detected_roles','detected_object_frames','object_roles','object_sample_indices','object_detection_indices',*process_keys]:np.testing.assert_array_equal(a[k],b[k])
            for e,pred in [('19',a),('20',b)]:
                m=models[e];r=m.score(d);visual,objects=independent_visual(m,d,*full_refs[e]);np.testing.assert_array_equal(r['visual'],visual);np.testing.assert_array_equal(objects,pred['object_scores'])
                for k in ['visual','combined',*process_keys]:np.testing.assert_array_equal(hold_scores(d['indices'],r[k],n),pred[k])
                for k in scores[e]:scores[e][k].append(pred[k])
            y.append(b['labels']);valid.append(hold_scores(d['indices'],d['relation_valid'],n).astype(bool))
            for role,frames,_ in models['20'].raw(d)[0]:
                routes=direct_routes(models['20'],d,role,frames)
                for route in [0,1]:
                    mask=routes==route;key=f'{role}:{route}';row=usage.setdefault(key,{'observations':0,'role_cdf_fallback':0});row['observations']+=int(mask.sum());row['role_cdf_fallback']+=int(mask.sum()) if (role,route) not in full_refs['20'][1] else 0
            for val,label in [(0,'normal'),(1,'anomaly')]:
                for k in ['visual','transition','dwell']:branch[k][label]+=int(np.sum((b['labels']==val)&(b[k]>normal_q['20'])))
                branch['process_only_over_visual'][label]+=int(np.sum((b['labels']==val)&(b['combined']>normal_q['20'])&~(b['visual']>normal_q['20'])))
    y=np.concatenate(y);valid=np.concatenate(valid);scores={e:{k:np.concatenate(v) for k,v in branches.items()} for e,branches in scores.items()};m=json.loads((out/'metrics.json').read_text());assert len(y)==8154
    for k,s in scores['20'].items():assert metric(y,s)==m['metrics'][k]
    before=scores['19']['combined']>normal_q['19'];after=scores['20']['combined']>normal_q['20'];assert np.mean(after[y==0])==m['test_normal_frame_alarm_rate'] and np.mean(after[y==1])==m['test_anomaly_frame_recall_at_q99']
    def counts(mask):return {'normal':int(np.sum(mask&(y==0))),'anomaly':int(np.sum(mask&(y==1)))}
    strata={}
    for observed in [False,True]:
        mask=valid==observed;strata[str(observed)]={'frames':int(mask.sum()),'normal_frames':int(np.sum(mask&(y==0))),'anomaly_frames':int(np.sum(mask&(y==1))),'alarms_before':counts(mask&before),'alarms_after':counts(mask&after),'added':counts(mask&after&~before),'removed':counts(mask&before&~after),'metrics':{e:{k:metric(y[mask],s[mask]) for k,s in branches.items()} for e,branches in scores.items()}}
    diagnostic={'normal_q99':normal_q,'added_alarms':counts(after&~before),'removed_alarms':counts(before&~after),'strata_relation_observed':strata,'branch_exceedances':branch,'test_route_usage_observations':usage,'note':'Each model uses its own normal q99. PCA, raw residuals, masks and process scores preserved. CDF path only changes; bank usage counts global/crop observations, not unique frames.'}
    (out/'route_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({'feature_sequences_checked':44,'normal_holdout_folds_checked':10,'test_sequences_checked':19,'protocol_hashes_match':True,'feature_arrays_byte_identical':True,'all_pca_and_raw_residuals_preserved':True,'independent_route_cdf_reconstruction':True,'holdout_exclusion_references_q99_verified':True,'raw_and_calibrated_process_unchanged':True,'legacy_experiment19_reproduced':True,'preflight_and_final_calibration_identical':True,'scores_and_source_labels_reproduced':True,'metrics_recomputed':True},indent=2)+'\n');print(json.dumps({k:v for k,v in diagnostic.items() if k not in ['strata_relation_observed','test_route_usage_observations']},indent=2))


if __name__=='__main__':main()
