"""Freeze all scores before opening test labels; retain per-process comparisons."""
import csv,json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.data import hold_scores,evaluation_labels
from ipad_vad.events import anomaly_events,summarize_events
from evaluate_baseline import metrics
from ipad_vad.learned_detector import sha,write
ART=Path("artifacts/experiment41");OUT=Path("results/experiment41")
from experiment41_normal import SCENES,RUNS,cfg,factory,load,arrays,verify_normal



def main():
    if (OUT/'metrics.json').exists():raise RuntimeError('Evaluation already completed; validate saved outputs instead')
    verify_normal();manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text());test=json.loads((OUT/'test_data_manifest.json').read_text());options=json.loads(Path('configs/experiment40_representation.json').read_text());options['source_features']={s:str(ART/'detections/features'/s) for s in SCENES};audit=json.loads((OUT/'normal_models_audit.json').read_text());files=[Path(__file__),OUT/'normal_models_checkpoint.json',OUT/'test_data_manifest.json'];prediction_files=[];process_checks=0;source_checks=0
    for run in RUNS:
        p=OUT/f'{run}_testing_extraction.json';record=json.loads(p.read_text());files.append(p);assert record['labels_opened'] is False
        for row in record['sequences']:
            path=ART/run/'features'/row['scene']/f'testing_{row["sequence"]}.npz';assert sha(path)==row['sha256'];files.append(path)
    protocol={'file_sha256':{str(p):sha(p) for p in files},'labels_opened':False};write(OUT/'pre_test_scoring_protocol.json',protocol)
    labels_allowed=False
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and '/test_label/' in str(args[0]) and not labels_allowed:raise RuntimeError('Labels forbidden until all scores are frozen')
    sys.addaudithook(guard)
    for scene in SCENES:
        split=manifest['subsplits'][scene];control={}
        for run in RUNS:
            model=factory(cfg(run,scene));model.fit([load(run,scene,s) for s in split['downstream_fit']]);model.calibrate([load(run,scene,s) for s in split['normal_calibration']])
            with np.load(ART/'normal_models'/f'{run}_{scene}_full_model.npz',allow_pickle=False) as f:expected=dict(f)
            actual=arrays(model);assert set(actual)==set(expected)
            for k,v in actual.items():np.testing.assert_array_equal(v,expected[k],err_msg=f'{run}/{scene}/{k}')
            saved_normal={}
            for seq in split['normal_calibration']:
                for k,v in model.score(load(run,scene,seq)).items():
                    if isinstance(v,np.ndarray):saved_normal.setdefault(k,[]).append(v)
            with np.load(ART/'normal_models'/f'{run}_{scene}_full_scores.npz',allow_pickle=False) as f:
                assert set(f)==set(saved_normal)
                for k,v in saved_normal.items():np.testing.assert_array_equal(np.concatenate(v),f[k])
            for item in [r for r in test['sequences'] if r['scene']==scene]:
                seq=item['sequence'];source=Path(options['source_features'][scene])/f'testing_{seq}.npz';assert sha(source)==test['test_source_feature_sha256'][str(source)]
                with np.load(source,allow_pickle=False) as f:original=dict(f)
                with np.load(ART/run/'features'/scene/source.name,allow_pickle=False) as f:data=dict(f)
                assert set(data)==set(original)
                for k,v in original.items():
                    if k not in ['global_features','crop_features']:np.testing.assert_array_equal(data[k],v)
                source_checks+=1;data['sequence_id']=f'{scene}/testing_{seq}';result=model.score(data);n=int(data['frame_count']);saved={k:hold_scores(data['indices'],v,n) for k,v in result.items() if isinstance(v,np.ndarray)}
                process_keys=[k for k in saved if k not in ['visual','combined']]
                if run=='A':control[seq]={k:saved[k] for k in process_keys}
                else:
                    assert set(process_keys)==set(control[seq])
                    for k in process_keys:np.testing.assert_array_equal(saved[k],control[seq][k]);process_checks+=1
                object_rows=[(role,int(frame),float(score)) for role,frames,scores in result['objects'] for frame,score in zip(frames,scores)]
                detection_ids=np.concatenate([np.full(len(frames),-1,dtype=int) if role==-1 else np.flatnonzero(data['roles']==role) for role,frames,scores in result['objects']])
                assert len(detection_ids)==len(object_rows)
                saved.update(indices=data['indices'],phases=data['phases'],threshold=np.array(model.threshold),objects=np.array(object_rows),object_detection_indices=detection_ids,boxes=data['boxes'],roles=data['roles'],tracks=data['tracks'],object_frames=data['object_frames'])
                path=ART/run/'predictions'/f'{scene}_{seq}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**saved);prediction_files.append(path)
            print('Scored without labels',scene,run,flush=True)
    checkpoint={'file_sha256':{str(p):sha(p) for p in prediction_files},'labels_opened':False,'source_metadata_checks':source_checks,'process_array_checks':process_checks};write(OUT/'test_scores_checkpoint.json',checkpoint)
    labels_allowed=True;rows=[];summaries=[];event_rows=[];label_files={};cache={};sequence_labels={}
    for scene in SCENES:
        for item in [r for r in test['sequences'] if r['scene']==scene]:
            seq=item['sequence'];p=Path(options['data_root'])/scene/'test_label'/f'{int(seq):03}.npy';raw=np.load(p);label_files[str(p)]=sha(p);y=evaluation_labels(raw,item['source_frames']);sequence_labels[(scene,seq)]=y
        for run in RUNS:
            ys=[];scores={k:[] for k in ['visual','process','combined']};events=[];q=None
            for item in [r for r in test['sequences'] if r['scene']==scene]:
                seq=item['sequence'];y=sequence_labels[(scene,seq)];ys.append(y)
                with np.load(ART/run/'predictions'/f'{scene}_{seq}.npz',allow_pickle=False) as f:d=dict(f)
                q=float(d['threshold']);alarm=d['combined']>q
                for k in scores:scores[k].append(d[k])
                ev=[dict(e,scene=scene,sequence=seq,run=run) for e in anomaly_events(y,alarm)];events.extend(ev);event_rows.extend(ev)
                row={'scene':scene,'run':run,'sequence':seq,'frames':len(y),'valid_frames':int((y>=0).sum()),'unknown_frames':int((y<0).sum()),'normal_frames':int((y==0).sum()),'anomaly_frames':int((y==1).sum()),'fp':int(np.sum(alarm&(y==0))),'tp':int(np.sum(alarm&(y==1))),'q99':q,'events':len(ev),'detected_events':sum(e['detected'] for e in ev)}
                for k in scores:
                    mm=metrics(y,d[k]);row[k+'_auroc']=mm['auroc'];row[k+'_ap']=mm['average_precision']
                rows.append(row)
            y=np.concatenate(ys);s={k:np.concatenate(v) for k,v in scores.items()};alarm=s['combined']>q;folds=[r for r in audit['folds'] if r['run']==run and r['scene']==scene];full=next(r for r in audit['full'] if r['run']==run and r['scene']==scene)
            summary={'run':run,'scene':scene,'metrics':{k:metrics(y,v) for k,v in s.items()},'normal_frames':int((y==0).sum()),'anomaly_frames':int((y==1).sum()),'unknown_frames':int((y<0).sum()),'q99':q,'fp':int(np.sum(alarm&(y==0))),'tp':int(np.sum(alarm&(y==1))),'normal_fpr':float(alarm[y==0].mean()),'anomaly_recall':float(alarm[y==1].mean()),'events':summarize_events(events),'normal_holdout_fp':sum(r['frame_alarms'] for r in folds),'normal_holdout_frames':sum(r['frames'] for r in folds),'normal_eligible':f'{run}_{scene}' in audit['eligible'],'subspaces':full['subspaces']};summaries.append(summary);cache[(scene,run)]=(y,s,q)
    macro=[]
    for run in RUNS:
        r=[s for s in summaries if s['run']==run];macro.append({'run':run,**{f'{branch}_{metric}':float(np.mean([s['metrics'][branch][metric] for s in r])) for branch in ['visual','combined'] for metric in ['auroc','average_precision']},'normal_fpr':float(np.mean([s['normal_fpr'] for s in r])),'anomaly_recall':float(np.mean([s['anomaly_recall'] for s in r])),'event_coverage':float(np.mean([s['events']['event_coverage'] for s in r]))})
    seed_summary=[]
    for arm in ['C','D']:
        r=[v for v in macro if v['run'].startswith(arm)];seed_summary.append({'arm':arm,'seeds':3,'macro':{k:{'mean':float(np.mean([v[k] for v in r])),'sample_std':float(np.std([v[k] for v in r],ddof=1))} for k in r[0] if k!='run'}})
    contrasts=[]
    for scene in SCENES:
        for run in RUNS:
            if run=='B':continue
            y,s,q=cache[(scene,run)];_,base,bq=cache[(scene,'B')];new=s['combined']>q;old=base['combined']>bq
            contrasts.append({'scene':scene,'run':run,'reference':'B','own_threshold_added_fp':int(np.sum(new&~old&(y==0))),'own_threshold_removed_fp':int(np.sum(~new&old&(y==0))),'own_threshold_added_tp':int(np.sum(new&~old&(y==1))),'own_threshold_removed_tp':int(np.sum(~new&old&(y==1))),'at_B_q99_fp':int(np.sum((s['combined']>bq)&(y==0))),'at_B_q99_tp':int(np.sum((s['combined']>bq)&(y==1)))})
    # Paired pre-41 frozen detector predictions use the same encoder/run, frames and labels.
    detector_contrasts=[]
    old_metrics=json.loads(Path('results/experiment40/metrics.json').read_text())
    old_checkpoint=json.loads(Path('results/experiment40/test_scores_checkpoint.json').read_text())
    for scene in SCENES:
        for run in RUNS:
            old_scores=[]
            for item in [r for r in test['sequences'] if r['scene']==scene]:
                path=Path('artifacts/experiment40')/run/'predictions'/f'{scene}_{item["sequence"]}.npz'
                assert sha(path)==old_checkpoint['file_sha256'][str(path)]
                with np.load(path,allow_pickle=False) as f:old_scores.append(f['combined'])
            y,s,q=cache[(scene,run)];old_score=np.concatenate(old_scores)
            ref=next(r for r in old_metrics['variants'] if r['scene']==scene and r['run']==run);newrow=next(r for r in summaries if r['scene']==scene and r['run']==run)
            bq=ref['q99'];old=old_score>bq;new=s['combined']>q
            detector_contrasts.append({'scene':scene,'run':run,'reference':'40_same_visual_frozen_detector','frozen_q99':bq,'learned_q99':q,
                'combined_auroc_delta':newrow['metrics']['combined']['auroc']-ref['metrics']['combined']['auroc'],
                'visual_auroc_delta':newrow['metrics']['visual']['auroc']-ref['metrics']['visual']['auroc'],
                'own_threshold_added_fp':int(np.sum(new&~old&(y==0))),'own_threshold_removed_fp':int(np.sum(~new&old&(y==0))),
                'own_threshold_added_tp':int(np.sum(new&~old&(y==1))),'own_threshold_removed_tp':int(np.sum(~new&old&(y==1))),
                'at_frozen_q99_fp':int(np.sum((s['combined']>bq)&(y==0))),'at_frozen_q99_tp':int(np.sum((s['combined']>bq)&(y==1)))})
    write(OUT/'metrics.json',{'variants':summaries,'scene_macro':macro,'adaptation_seed_summary':seed_summary,'paired_B_contrasts':contrasts,'paired_frozen_detector_contrasts':detector_contrasts,'label_sha256':label_files,'note':'Macro averages of within-process metrics; one shared detector seed, 3 prior visual seeds descriptive only. Comparison adds normal weak box supervision. Phase parameters fixed; statistical references refitted. All scores frozen before labels. Separate normal q99 for each run/process; fixed-B-threshold contrasts are secondary. Unknown label-length videos excluded in entirety. Reused development scenes are not independent confirmation.'})
    write(OUT/'events.json',event_rows)
    with (OUT/'per_sequence.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    for p,h in checkpoint['file_sha256'].items():assert sha(p)==h
    write(OUT/'evaluation_audit.json',{'models_refitted_exact_to_frozen_normal':32,'full_normal_scores_reconstructed':32,'source_metadata_checks':source_checks,'test_process_array_checks':process_checks,'prediction_files_frozen_before_any_labels':len(prediction_files),'predictions_unchanged_after_labels':True,'metric_rows':len(summaries),'per_sequence_rows':len(rows)})
    print(json.dumps(macro,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
