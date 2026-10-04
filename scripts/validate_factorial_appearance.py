"""Reconstruct four-cell appearance banks, normal holdout and all predictions."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from ipad_vad.scoring import observations,Subspace
from ipad_vad.data import hold_scores,evaluation_labels
from evaluate_route_holdout import load_cache,normal_model_arrays
from audit_factorial_normal import VARIANTS,bank_usage

CELLS={'18':(0,0),'21_fit':(1,0),'21_infer':(0,1),'19':(1,1)}
PROCESS=['transition','dwell','process','dwell_valid','dwell_age','dwell_reason','dwell_entry_context']


def load(p):
    with np.load(p,allow_pickle=False) as f:return dict(f)


def metric(y,s):return {'frames':len(y),'positive_frames':int(y.sum()),'auroc':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,'average_precision':float(average_precision_score(y,s)) if np.any(y==1) else None}


def direct_raw(m,d,b):
    result=[]
    for role,frames,x in observations(d):
        residual=np.empty(len(x))
        for p in np.unique(d['phases'][frames]):
            for observed in [False,True]:
                mask=(d['phases'][frames]==p)&(d['relation_valid'][frames]==observed)
                if not mask.any():continue
                key=(role,-1) if b and not observed else (role,int(p))
                if key not in m.spaces:key=(role,-1)
                if key not in m.spaces:key=(-1,-1)
                residual[mask]=m.spaces[key].residual(x[mask])
        result.append((role,frames,residual))
    return result


def verify_calibration(m,caches,b):
    expected={}
    for d in caches:
        for role,_,r in direct_raw(m,d,b):expected.setdefault(role,[]).append(r)
    for role,parts in expected.items():np.testing.assert_allclose(np.concatenate(parts),m.calibration[role],rtol=1e-12,atol=1e-12)
    assert m.route_calibration is None
    assert m.threshold==np.quantile(np.concatenate([m.score(d)['combined'] for d in caches]),.99,method='higher')


def main():
    out=Path('results/experiment21');art=Path('artifacts/experiment21');root=art/'features/R04';source=Path('artifacts/experiment19/features/R04');split=json.loads(Path('results/stage00/splits.json').read_text())['R04']
    protocol=json.loads((out/'pre_normal_protocol.json').read_text());checkpoint=json.loads((out/'pre_test_checkpoint.json').read_text())
    for record in [protocol,checkpoint]:
        for p,h in record['file_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    paths=sorted(source.glob('*.npz'));assert len(paths)==44
    for p in paths:
        h=protocol['source_features_sha256'][p.name];assert hashlib.sha256(p.read_bytes()).hexdigest()==h
        for e in ['21','21_fit','21_infer']:assert hashlib.sha256((Path(f'artifacts/experiment{e}/features/R04')/p.name).read_bytes()).hexdigest()==h
    fit=[load_cache(root,s) for s in split['fit']];cal={s:load_cache(root,s) for s in split['calibration']};holdout=json.loads((out/'normal_holdout.json').read_text());audit=json.loads((out/'normal_audit.json').read_text());models={};metrics={e:json.loads(Path(f'results/experiment{e}/metrics.json').read_text()) for e in VARIANTS};usage={e:{} for e in VARIANTS}
    with threadpool_limits(limits=4):
        for e in VARIANTS:
            cfg=json.loads(Path(f'configs/experiment{e}.json').read_text());assert 'appearance_route_calibration' not in cfg;m=LognormalDwellBaseline(cfg,load_process(cfg));m.fit(fit);models[e]=m
        for a,representative in [(0,'18'),(1,'19')]:
            m=models[representative];buckets={}
            for d in fit:
                for role,frames,x in observations(d):
                    buckets.setdefault((role,-1),[]).append(x)
                    for phase in range(m.k):
                        keep=d['phases'][frames]==phase
                        if a:keep&=d['relation_valid'][frames]
                        if keep.any():buckets.setdefault((role,phase),[]).append(x[keep])
            expected={k:np.concatenate(v) for k,v in buckets.items() if sum(len(x) for x in v)>=10};assert set(expected)==set(m.spaces)
            for k,x in expected.items():
                sub=Subspace(x,.95,32);assert m.spaces[k].n==len(x)
                for name in ['mean','basis']:np.testing.assert_array_equal(getattr(sub,name),getattr(m.spaces[k],name))
        for e,m in models.items():
            ref=models['19' if CELLS[e][0] else '18'];assert set(m.spaces)==set(ref.spaces);saved=load(f'artifacts/experiment{e}/normal_model.npz')
            for k,space in m.spaces.items():
                for name in ['mean','basis']:
                    np.testing.assert_array_equal(getattr(space,name),getattr(ref.spaces[k],name));np.testing.assert_array_equal(getattr(space,name),saved[f'{name}_{k[0]}_{k[1]}'])
                    if k[1]==-1:np.testing.assert_array_equal(getattr(space,name),getattr(models['18'].spaces[k],name))
            np.testing.assert_array_equal(m.transition,models['18'].transition)
        assert len(holdout['folds'])==20
        process_by_fold={}
        for row in holdout['folds']:
            e=row['variant'];held=row['held_out_sequence'];used=[s for s in split['calibration'] if s!=held];assert used==row['calibration_sequences'];m=copy.deepcopy(models[e]);m.calibrate([cal[s] for s in used]);verify_calibration(m,[cal[s] for s in used],CELLS[e][1]);refs=normal_model_arrays(m);saved=load(art/'normal_holdout'/f'{e}_exclude_{held}_references.npz');assert refs.keys()==saved.keys()
            for k,v in refs.items():np.testing.assert_array_equal(v,saved[k])
            d=cal[held];r=m.score(d);pred=load(art/'normal_holdout'/f'{e}_exclude_{held}_scores.npz')
            for k in ['visual','combined',*PROCESS]:np.testing.assert_array_equal(r[k],pred[k])
            np.testing.assert_array_equal(np.concatenate([x for _,_,x in r['objects']]),pred['object_scores'])
            if held in process_by_fold:
                for k in PROCESS:np.testing.assert_array_equal(r[k],process_by_fold[held][k])
            else:process_by_fold[held]=r
            assert m.threshold==row['normal_q99'];alarm=r['combined']>m.threshold;dense=hold_scores(d['indices'],alarm,int(d['frame_count'])).astype(bool);valid=hold_scores(d['indices'],d['relation_valid'],len(dense)).astype(bool)
            assert int(alarm.sum())==row['held_out_sample_alarms'] and len(alarm)==row['held_out_samples'];assert int(dense.sum())==row['held_out_frame_alarms'] and len(dense)==row['held_out_frames']
            for observed in [False,True]:
                mask=valid==observed;assert row['relation_strata'][str(observed)]=={'frames':int(mask.sum()),'alarms':int(np.sum(mask&dense))}
            for kind,name in [(0,'pooled'),(1,'phase')]:
                mask=hold_scores(d['indices'],m.appearance_routes(d,-1,np.arange(len(d['indices']))),len(dense))==kind;assert row['global_bank_strata'][name]=={'frames':int(mask.sum()),'alarms':int(np.sum(mask&dense))}
        for e in VARIANTS:
            rows=[r for r in holdout['folds'] if r['variant']==e]
            for k in ['held_out_samples','held_out_sample_alarms','held_out_frames','held_out_frame_alarms','held_out_at_one_samples']:assert holdout['totals'][e][k]==sum(r[k] for r in rows)
            m=models[e];m.calibrate(list(cal.values()));verify_calibration(m,list(cal.values()),CELLS[e][1]);saved=load(f'artifacts/experiment{e}/normal_calibration_scores.npz');pre=load(art/'full_normal'/f'{e}_scores.npz')
            for k,v in pre.items():np.testing.assert_array_equal(v,saved[k])
            for k in ['visual','combined',*PROCESS]:np.testing.assert_array_equal(np.concatenate([m.score(d)[k] for d in cal.values()]),saved[k])
            assert m.threshold==metrics[e]['normal_q99_threshold']==audit['variants'][e]['normal_q99']
        # Check inference/calibration bank selection across all 44 sequences.
        for p in paths:
            part,seq=p.stem.split('_');d=load_cache(root,seq,part);baseline_process=models['18'].raw(d)[1]
            for e,m in models.items():
                actual,process=m.raw(d);np.testing.assert_array_equal(process,baseline_process)
                for (r,f,x),(rr,ff,xx) in zip(actual,direct_raw(m,d,CELLS[e][1])):assert r==rr;np.testing.assert_array_equal(f,ff);np.testing.assert_allclose(x,xx,rtol=1e-12,atol=1e-12)
                group='test' if part=='testing' else 'fit' if seq in split['fit'] else 'calibration';counts=usage[e].setdefault(group,{})
                for k,v in bank_usage(m,d).items():counts[k]=counts.get(k,0)+v
        y=[];valid=[];scores={e:{k:[] for k in ['visual','process','combined']} for e in VARIANTS};branches={e:{k:{'normal':0,'anomaly':0} for k in ['visual','transition','dwell','process_only_over_visual']} for e in VARIANTS}
        test_paths=sorted(root.glob('testing_*.npz'));assert len(test_paths)==19
        for p in test_paths:
            seq=p.stem.split('_')[1];d=load_cache(root,seq,'testing');n=int(d['frame_count']);labels=evaluation_labels(np.load(Path('/media/jeong/ExtHDD/IPAD_vLLM/IPAD_dataset/R04/test_label')/f'{int(seq):03}.npy'),n);y.append(labels);valid.append(hold_scores(d['indices'],d['relation_valid'],n).astype(bool));reference=None
            for e,m in models.items():
                saved=load(f'artifacts/experiment{e}/predictions/R04_{seq}.npz');r=m.score(d);np.testing.assert_array_equal(saved['labels'],labels)
                for k in ['visual','combined',*PROCESS]:np.testing.assert_array_equal(hold_scores(d['indices'],r[k],n),saved[k])
                np.testing.assert_array_equal(np.concatenate([x for _,_,x in r['objects']]),saved['object_scores'])
                if reference is None:reference=saved
                else:
                    for k in ['labels','indices','phases','boxes','tracks','detected_roles','detected_object_frames','object_roles','object_sample_indices','object_detection_indices',*PROCESS]:np.testing.assert_array_equal(saved[k],reference[k])
                for k in scores[e]:scores[e][k].append(saved[k])
                for val,name in [(0,'normal'),(1,'anomaly')]:
                    for k in ['visual','transition','dwell']:branches[e][k][name]+=int(np.sum((labels==val)&(saved[k]>m.threshold)))
                    branches[e]['process_only_over_visual'][name]+=int(np.sum((labels==val)&(saved['combined']>m.threshold)&~(saved['visual']>m.threshold)))
    y=np.concatenate(y);valid=np.concatenate(valid);assert len(y)==8154;scores={e:{k:np.concatenate(v) for k,v in s.items()} for e,s in scores.items()};alarms={e:s['combined']>models[e].threshold for e,s in scores.items()}
    def counts(mask):return {'normal':int(np.sum(mask&(y==0))),'anomaly':int(np.sum(mask&(y==1)))}
    strata={};summary={}
    for e in VARIANTS:
        for k,s in scores[e].items():assert metric(y,s)==metrics[e]['metrics'][k]
        assert np.mean(alarms[e][y==0])==metrics[e]['test_normal_frame_alarm_rate'] and np.mean(alarms[e][y==1])==metrics[e]['test_anomaly_frame_recall_at_q99']
        summary[e]={'A':CELLS[e][0],'B':CELLS[e][1],'alarms':counts(alarms[e]),'metrics':metrics[e]['metrics'],'q99':models[e].threshold};strata[e]={}
        for observed in [False,True]:
            mask=valid==observed;strata[e][str(observed)]={'frames':int(mask.sum()),'normal_frames':int(np.sum(mask&(y==0))),'anomaly_frames':int(np.sum(mask&(y==1))),'alarms':counts(mask&alarms[e]),'metrics':{k:metric(y[mask],s[mask]) for k,s in scores[e].items()}}
    pairs={}
    for name,a,b in [('A_when_B0','18','21_fit'),('A_when_B1','21_infer','19'),('B_when_A0','18','21_infer'),('B_when_A1','21_fit','19')]:
        added=alarms[b]&~alarms[a];removed=alarms[a]&~alarms[b];pairs[name]={'before':a,'after':b,'added':counts(added),'removed':counts(removed),'delta_combined_auroc':metrics[b]['metrics']['combined']['auroc']-metrics[a]['metrics']['combined']['auroc'],'relation_strata':{str(v):{'added':counts(added&(valid==v)),'removed':counts(removed&(valid==v))} for v in [False,True]}}
    (out/'factorial_diagnostic.json').write_text(json.dumps({'cells':summary,'strata_relation_observed':strata,'factor_contrasts':pairs,'bank_usage_observations':usage,'branch_exceedances':branches,'note':'All four cells disclosed; each refits role CDF and normal q99. Descriptive repeated R04 development comparison, not independent factorial significance testing.'},indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({'feature_sequences_checked':44,'normal_holdout_cells_checked':20,'test_predictions_checked':76,'protocol_hashes_match':True,'feature_arrays_byte_identical':True,'banks_independently_reconstructed':True,'same_A_same_pca_and_all_pooled_preserved':True,'all_raw_routing_checked':True,'holdout_exclusion_references_q99_verified':True,'process_scores_preserved_all_cells':True,'legacy_18_19_reproduced':True,'normal_preflight_matches_final':True,'scores_and_source_labels_reproduced':True,'all_metrics_recomputed':True},indent=2)+'\n');print(json.dumps({'summary':{e:{k:v for k,v in r.items() if k!='metrics'} for e,r in summary.items()},'contrasts':pairs},indent=2))


if __name__=='__main__':main()
