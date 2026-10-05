"""Reconstruct normal holdouts and independently recompute rank/threshold metrics."""
import copy,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from ipad_vad.data import hold_scores,evaluation_labels
from ipad_vad.events import anomaly_events,summarize_events
from candidates45_common import ART,OUT,sha,write,config,verify_record
from experiment45_normal import SCENES,RUNS,cfg,factory,load,arrays,verify_normal


def independent_metrics(y,s):
    valid=y>=0;y=y[valid];s=s[valid];n1=int(y.sum());n0=len(y)-n1
    auc=float((rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else None
    order=np.argsort(-s,kind='stable');ys=y[order];ss=s[order];ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1];tp=np.cumsum(ys)[ends];count=ends+1
    ap=float(np.sum(np.diff(np.r_[0,tp])/n1*(tp/count))) if n1 else None
    return auc,ap


def main():
    verify_normal();result=json.loads((OUT/'metrics.json').read_text());manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text());test=json.loads(Path('results/experiment40/test_data_manifest.json').read_text());audit=json.loads((OUT/'normal_models_audit.json').read_text());options=json.loads(Path('configs/experiment40_representation.json').read_text());folds_checked=0;predictions_checked=0;normal_feature_files=0
    for name in ['probe_protocol','selection_checkpoint','pre_test_scoring_protocol','test_scores_checkpoint']:
        for path,h in json.loads((OUT/f'{name}.json').read_text())['file_sha256'].items():assert sha(path)==h,path
    for path,h in result['label_sha256'].items():assert sha(path)==h
    probe_checks=0
    probe=json.loads((ART/'probe_views.json').read_text())
    for part,rows in probe.items():
        key='representation_train' if part=='train' else 'representation_validation'
        for r in rows:
            split=manifest['subsplits'][r['scene']]
            assert r['sequence'] in split[key] and r['sequence'] not in split['normal_calibration']
    records=[]
    for candidate in config()['candidates']:
        record=json.loads((OUT/f'{candidate}_probe.json').read_text());assert sha(record['cache_path'])==record['cache_sha256'];records.append(record)
        with np.load(record['cache_path'],allow_pickle=False) as f:
            for row in record['rows']:
                residuals=f[row['scene']+'_'+row['kind']+'_residuals']
                j=['occlusion','local_shuffle','local_noise'].index(row['corruption'])+2
                neg=residuals[:,:2].reshape(-1);pos=residuals[:,j]
                auc,_=independent_metrics(np.r_[np.zeros(len(neg)),np.ones(len(pos))],np.r_[neg,pos])
                np.testing.assert_allclose(auc,row['auroc'],rtol=0,atol=1e-12);probe_checks+=1
        np.testing.assert_allclose(record['selection_score'],np.mean([r['auroc'] for r in record['rows']]),rtol=0,atol=0)
    best=max(r['selection_score'] for r in records)
    eligible=[r for r in records if r['selection_score']>=best-.005]
    selected=min(eligible,key=lambda r:(r['encoder_seconds']/r['encoded_views'],r['candidate']))['candidate']
    assert selected==json.loads((OUT/'selection.json').read_text())['selected']
    for scene in SCENES:
        split=manifest['subsplits'][scene];assert set(split['representation_train']).isdisjoint(split['representation_validation']);assert sorted(split['representation_train']+split['representation_validation'])==sorted(split['downstream_fit']);assert set(split['downstream_fit']).isdisjoint(split['normal_calibration'])
        labels={}
        for row in [r for r in test['sequences'] if r['scene']==scene]:
            p=Path(options['data_root'])/scene/'test_label'/f'{int(row["sequence"]):03}.npy';labels[row['sequence']]=evaluation_labels(np.load(p),row['source_frames'])
        for run in RUNS:
            record=json.loads(((Path('results/experiment40') if run in config()['controls'] else OUT)/f'{run}_normal_extraction.json').read_text())
            for row in [r for r in record['sequences'] if r['scene']==scene]:
                p=ART/run/'features'/scene/f'training_{row["sequence"]}.npz';assert sha(p)==row['sha256'];normal_feature_files+=1
            base=factory(cfg(run,scene));base.fit([load(run,scene,s) for s in split['downstream_fit']]);cal={s:load(run,scene,s) for s in split['normal_calibration']}
            for held in [None,*cal]:
                m=copy.deepcopy(base);m.calibrate([v for s,v in cal.items() if s!=held]);suffix=f'{run}_{scene}_{held or "full"}'
                with np.load(ART/'normal_models'/f'{suffix}_model.npz',allow_pickle=False) as f:
                    actual=arrays(m);assert set(f)==set(actual)
                    for k,v in actual.items():np.testing.assert_array_equal(v,f[k])
                probe={s:v for s,v in cal.items() if held is None or s==held};rescored={s:m.score(v) for s,v in probe.items()};fp=sum(int(hold_scores(probe[s]['indices'],v['combined']>m.threshold,int(probe[s]['frame_count'])).sum()) for s,v in rescored.items())
                with np.load(ART/'normal_models'/f'{suffix}_scores.npz',allow_pickle=False) as f:
                    for k in f:np.testing.assert_array_equal(np.concatenate([v[k] for v in rescored.values()]),f[k])
                row=next(r for r in (audit['full'] if held is None else audit['folds']) if r['scene']==scene and r['run']==run and r['held_out']==held);assert fp==row['frame_alarms'] and m.threshold==row['q99'];folds_checked+=1
                if held is None:full=m
            y=[];s={k:[] for k in ['visual','process','combined']};events=[]
            for seq,lab in labels.items():
                with np.load(ART/run/'features'/scene/f'testing_{seq}.npz',allow_pickle=False) as f:data=dict(f)
                data['sequence_id']=f'{scene}/testing_{seq}';rescored=full.score(data)
                with np.load(ART/run/'predictions'/f'{scene}_{seq}.npz',allow_pickle=False) as f:saved=dict(f)
                for k,v in rescored.items():
                    if isinstance(v,np.ndarray):np.testing.assert_array_equal(hold_scores(data['indices'],v,len(lab)),saved[k])
                objects=np.array([(role,int(frame),float(score)) for role,frames,scores in rescored['objects'] for frame,score in zip(frames,scores)])
                np.testing.assert_array_equal(objects,saved['objects'])
                detection_ids=np.concatenate([np.full(len(frames),-1,dtype=int) if role==-1 else np.flatnonzero(data['roles']==role) for role,frames,scores in rescored['objects']])
                np.testing.assert_array_equal(detection_ids,saved['object_detection_indices'])
                for k in s:s[k].append(saved[k])
                y.append(lab);events.extend(anomaly_events(lab,saved['combined']>full.threshold));predictions_checked+=1
            y=np.concatenate(y);s={k:np.concatenate(v) for k,v in s.items()};row=next(r for r in result['variants'] if r['scene']==scene and r['run']==run)
            for k,v in s.items():
                auc,ap=independent_metrics(y,v);np.testing.assert_allclose([auc,ap],[row['metrics'][k]['auroc'],row['metrics'][k]['average_precision']],rtol=1e-12,atol=1e-12)
            alarm=s['combined']>full.threshold;assert int(np.sum(alarm&(y==0)))==row['fp'] and int(np.sum(alarm&(y==1)))==row['tp'];assert summarize_events(events)==row['events'];print('Validated',scene,run,flush=True)
    for r in result['scene_macro']:
        rows=[v for v in result['variants'] if v['run']==r['run']]
        for branch in ['visual','combined']:
            for metric in ['auroc','average_precision']:np.testing.assert_allclose(r[f'{branch}_{metric}'],np.mean([v['metrics'][branch][metric] for v in rows]),rtol=0,atol=0)
    write(OUT/'validation.json',{'normal_files_verified':normal_feature_files,'full_and_holdout_normal_models_scores_reconstructed':folds_checked,'test_predictions_reconstructed':predictions_checked,'auroc_recomputed_from_midranks':True,'ap_recomputed_from_tied_thresholds':True,'alarms_events_recomputed':True,'protocols_verified':True,'new_training_runs':0,'probe_aurocs_independently_recomputed':probe_checks,'selected_candidate':selected,'selection_normal_only':True,'independent_recording_groups_available':False})

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
