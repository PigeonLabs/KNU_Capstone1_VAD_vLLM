"""Rebuild experiment23 common ranks, excluded-video calibration and all paired predictions."""
import copy,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from ipad_vad.data import hold_scores,evaluation_labels
from ipad_vad.events import anomaly_events,summarize_events
from evaluate_route_holdout import load_cache,normal_model_arrays
from audit_common_rank import VARIANTS,BEFORE,AFTER,PAIRS,PROCESS,verify_pair,verify_raw_pair,sha
from audit_factorial_normal import bank_usage
from validate_factorial_appearance import load,metric,direct_raw,verify_calibration


def main():
    out=Path('results/experiment23');art=Path('artifacts/experiment23');root=art/'features/R04';split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];protocol=json.loads((out/'pre_normal_protocol.json').read_text())
    for record in [protocol,json.loads((out/'pre_test_checkpoint.json').read_text())]:
        for p,h in record['file_sha256'].items():assert sha(p)==h,p
    assert len(protocol['source_features_sha256'])==44
    for p,h in protocol['source_features_sha256'].items():
        for e in ['23',*AFTER]:assert sha(Path(f'artifacts/experiment{e}/features/R04')/p)==h
    fit=[load_cache(root,s) for s in split['fit']];cal={s:load_cache(root,s) for s in split['calibration']};audit=json.loads((out/'normal_audit.json').read_text());holdout=json.loads((out/'normal_holdout.json').read_text());models={};metrics={e:json.loads(Path(f'results/experiment{e}/metrics.json').read_text()) for e in VARIANTS};usage={e:{} for e in VARIANTS};events={e:[] for e in VARIANTS};scores={e:{k:[] for k in ['visual','process','combined']} for e in VARIANTS};y=[];valid=[];by_sequence={e:{} for e in VARIANTS}
    with threadpool_limits(limits=4):
        for e in VARIANTS:
            cfg=json.loads(Path(f'configs/experiment{e}.json').read_text());m=LognormalDwellBaseline(cfg,load_process(cfg));m.fit(fit);models[e]=m;saved=load(f'artifacts/experiment{e}/normal_model.npz');selected=load(art/'full_normal'/f'{e}_selections.npz')
            for k,space in m.spaces.items():
                for name in ['mean','basis']:
                    np.testing.assert_array_equal(getattr(space,name),saved[f'{name}_{k[0]}_{k[1]}'])
                    if k[1]==-1:np.testing.assert_array_equal(getattr(space,name),getattr(models['21_fit'].spaces[k],name))
            for (r,p),idx in m.sampling_indices.items():
                np.testing.assert_array_equal(idx,saved[f'fit_selection_{r}_{p}']);np.testing.assert_array_equal(idx,selected[f'{r}_{p}'])
            if e in AFTER:
                assert set(m.spaces)==set(models['21_fit'].spaces)
                for k,s in m.spaces.items():assert s.n==models['21_fit'].spaces[k].n
            np.testing.assert_array_equal(m.transition,models['21_fit'].transition)
        for after,before in PAIRS.items():
            rows,_=verify_pair(models[before],models[after],fit,before);assert rows==audit['pairs'][after]['banks'];assert verify_raw_pair(models[before],models[after],fit+list(cal.values()))==audit['pairs'][after]['normal_raw_checks']
        common={f'{r}:{p}':min(models[e].spaces[r,p].rank for e in BEFORE) for r,p in models[BEFORE[0]].spaces if p>=0};assert common==audit['common_ranks']
        for e in AFTER:assert models[e].cfg['appearance_phase_ranks']==common
        assert len(holdout['folds'])==60;process_by_fold={}
        for row in holdout['folds']:
            e=row['variant'];held=row['held_out_sequence'];used=[s for s in split['calibration'] if s!=held];assert row['calibration_sequences']==used and held not in used;m=copy.deepcopy(models[e]);m.calibrate([cal[s] for s in used]);verify_calibration(m,[cal[s] for s in used],0)
            refs=normal_model_arrays(m);saved=load(art/'normal_holdout'/f'{e}_exclude_{held}_references.npz');assert refs.keys()==saved.keys()
            for k,v in refs.items():np.testing.assert_array_equal(v,saved[k])
            d=cal[held];r=m.score(d);pred=load(art/'normal_holdout'/f'{e}_exclude_{held}_scores.npz')
            for k in ['visual','combined',*PROCESS]:np.testing.assert_array_equal(r[k],pred[k])
            np.testing.assert_array_equal(np.concatenate([s for _,_,s in r['objects']]),pred['object_scores'])
            if held in process_by_fold:
                for k in PROCESS:np.testing.assert_array_equal(r[k],process_by_fold[held][k])
            else:process_by_fold[held]=r
            assert m.threshold==row['normal_q99'];alarm=r['combined']>m.threshold;dense=hold_scores(d['indices'],alarm,int(d['frame_count'])).astype(bool);obs=hold_scores(d['indices'],d['relation_valid'],len(dense)).astype(bool)
            assert int(alarm.sum())==row['held_out_sample_alarms'] and len(alarm)==row['held_out_samples'];assert int(dense.sum())==row['held_out_frame_alarms'] and len(dense)==row['held_out_frames']
            for v in [False,True]:assert row['relation_strata'][str(v)]=={'frames':int(np.sum(obs==v)),'alarms':int(np.sum(dense&(obs==v)))}
        for e,m in models.items():
            rows=[r for r in holdout['folds'] if r['variant']==e]
            for k in ['held_out_samples','held_out_sample_alarms','held_out_frames','held_out_frame_alarms','held_out_at_one_samples']:assert sum(r[k] for r in rows)==holdout['totals'][e][k]
            m.calibrate(list(cal.values()));verify_calibration(m,list(cal.values()),0);saved=load(f'artifacts/experiment{e}/normal_calibration_scores.npz');pre=load(art/'full_normal'/f'{e}_scores.npz')
            for k,v in pre.items():np.testing.assert_array_equal(saved[k],v)
            for k in ['visual','combined',*PROCESS]:np.testing.assert_array_equal(np.concatenate([m.score(d)[k] for d in cal.values()]),saved[k])
            assert m.threshold==metrics[e]['normal_q99_threshold']==audit['variants'][e]['normal_q99']
        for p in sorted(root.glob('*.npz')):
            part,seq=p.stem.split('_');d=load_cache(root,seq,part);group='test' if part=='testing' else 'fit' if seq in split['fit'] else 'calibration'
            for e,m in models.items():
                for (r,f,x),(rr,ff,xx) in zip(m.raw(d)[0],direct_raw(m,d,0)):
                    assert r==rr;np.testing.assert_array_equal(f,ff);np.testing.assert_allclose(x,xx,rtol=1e-12,atol=1e-12)
                counts=usage[e].setdefault(group,{})
                for k,v in bank_usage(m,d).items():counts[k]=counts.get(k,0)+v
        raw_pairs={after:verify_raw_pair(models[before],models[after],[load_cache(root,p.stem.split('_')[1],'testing') for p in sorted(root.glob('testing_*.npz'))]) for after,before in PAIRS.items()}
        for p in sorted(root.glob('testing_*.npz')):
            seq=p.stem.split('_')[1];d=load_cache(root,seq,'testing');n=int(d['frame_count']);labels=evaluation_labels(np.load(Path('/media/jeong/ExtHDD/IPAD_vLLM/IPAD_dataset/R04/test_label')/f'{int(seq):03}.npy'),n);y.append(labels);valid.append(hold_scores(d['indices'],d['relation_valid'],n).astype(bool));ref=None
            for e,m in models.items():
                saved=load(f'artifacts/experiment{e}/predictions/R04_{seq}.npz');r=m.score(d);np.testing.assert_array_equal(labels,saved['labels'])
                for k in ['visual','combined',*PROCESS]:np.testing.assert_array_equal(hold_scores(d['indices'],r[k],n),saved[k])
                np.testing.assert_array_equal(np.concatenate([s for _,_,s in r['objects']]),saved['object_scores'])
                if ref is None:ref=saved
                else:
                    for k in ['labels','indices','phases','boxes','tracks','detected_roles','detected_object_frames',*PROCESS]:np.testing.assert_array_equal(saved[k],ref[k])
                for k in scores[e]:scores[e][k].append(saved[k])
                alarm=saved['combined']>m.threshold;events[e].extend([dict(v,sequence_key=f'R04_{seq}') for v in anomaly_events(labels,alarm)]);by_sequence[e][seq]={'normal':int(np.sum(alarm&(labels==0))),'anomaly':int(np.sum(alarm&(labels==1)))}
    y=np.concatenate(y);valid=np.concatenate(valid);assert len(y)==8154;scores={e:{k:np.concatenate(v) for k,v in s.items()} for e,s in scores.items()};alarms={e:s['combined']>models[e].threshold for e,s in scores.items()}
    def counts(mask):return {'normal':int(np.sum(mask&(y==0))),'anomaly':int(np.sum(mask&(y==1)))}
    summary={};strata={};pairs={}
    for e in VARIANTS:
        for k,s in scores[e].items():assert metric(y,s)==metrics[e]['metrics'][k]
        assert float(np.mean(alarms[e][y==0]))==metrics[e]['test_normal_frame_alarm_rate'];assert float(np.mean(alarms[e][y==1]))==metrics[e]['test_anomaly_frame_recall_at_q99']
        summary[e]={'q99':models[e].threshold,'alarms':counts(alarms[e]),'normal_fpr':float(np.mean(alarms[e][y==0])),'anomaly_recall':float(np.mean(alarms[e][y==1])),'metrics':metrics[e]['metrics'],'events':summarize_events(events[e]),'normal_holdout_fp':holdout['totals'][e]['held_out_frame_alarms']};strata[e]={}
        for v in [False,True]:
            mask=valid==v;strata[e][str(v)]={'frames':int(mask.sum()),'normal_frames':int(np.sum(mask&(y==0))),'anomaly_frames':int(np.sum(mask&(y==1))),'alarms':counts(mask&alarms[e]),'metrics':{k:metric(y[mask],s[mask]) for k,s in scores[e].items()}}
        assert usage[e]==usage['21_fit']
    for name,before,after in [(f'rank_{after}',before,after) for after,before in PAIRS.items()]+[(f'selection_{e}','23_obs',e) for e in AFTER[1:]]:
        pairs[name]={'before':before,'after':after,'added':counts(alarms[after]&~alarms[before]),'removed':counts(alarms[before]&~alarms[after]),'delta_visual_auroc':summary[after]['metrics']['visual']['auroc']-summary[before]['metrics']['visual']['auroc'],'delta_combined_auroc':summary[after]['metrics']['combined']['auroc']-summary[before]['metrics']['combined']['auroc'],'gained_events':[],'lost_events':[]}
        for a,b in zip(events[before],events[after]):
            identity={k:a[k] for k in ['sequence_key','start_frame','end_frame_exclusive']};assert identity=={k:b[k] for k in identity}
            if b['detected'] and not a['detected']:pairs[name]['gained_events'].append(identity)
            if a['detected'] and not b['detected']:pairs[name]['lost_events'].append(identity)
    values={'combined_auroc':[summary[e]['metrics']['combined']['auroc'] for e in AFTER[1:]],'combined_ap':[summary[e]['metrics']['combined']['average_precision'] for e in AFTER[1:]],'visual_auroc':[summary[e]['metrics']['visual']['auroc'] for e in AFTER[1:]],'test_fp':[summary[e]['alarms']['normal'] for e in AFTER[1:]],'test_tp':[summary[e]['alarms']['anomaly'] for e in AFTER[1:]],'events':[summary[e]['events']['detected_events'] for e in AFTER[1:]],'normal_holdout_fp':[summary[e]['normal_holdout_fp'] for e in AFTER[1:]]}
    ranges={k:{'mean':float(np.mean(v)),'min':float(min(v)),'max':float(max(v))} for k,v in values.items()}
    (out/'common_rank_diagnostic.json').write_text(json.dumps({'variants':summary,'random_seed_descriptive_ranges':ranges,'strata_relation_observed':strata,'paired_contrasts':pairs,'test_raw_rank_checks':raw_pairs,'bank_usage_observations':usage,'per_sequence_alarms':by_sequence,'events':events,'note':'Six fixed normal FIT selections and common per-bank ranks. All legacy/rank-controlled pairs disclosed. Normal CDF/q99 refitted per configuration; B off. Seed ranges on repeated R04 development data are descriptive, not confidence intervals.'},indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({'feature_sequences_checked':44,'normal_holdout_configurations_checked':60,'test_predictions_checked':228,'protocol_hashes_match':True,'features_byte_identical':True,'normal_fit_only_counts_indices_unique_reproduced':True,'common_rank_minimum_from_normal_fit':True,'paired_means_prefix_basis_and_raw_residual_monotonicity_verified':True,'banks_independently_reconstructed':True,'all_pooled_preserved':True,'matched_bank_support_counts_routes':True,'holdout_exclusion_references_q99_verified':True,'process_scores_preserved_all_variants':True,'all_six_legacy_variants_reproduced':True,'normal_preflight_matches_final':True,'scores_source_labels_and_metrics_recomputed':True},indent=2)+'\n');print(json.dumps({'ranges':ranges,'summary':{e:{k:v for k,v in r.items() if k!='metrics'} for e,r in summary.items()}},indent=2))


if __name__=='__main__':main()
