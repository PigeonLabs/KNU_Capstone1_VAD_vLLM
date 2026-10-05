"""Freeze all association-branch scores before evaluation labels are opened."""
import csv,json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.data import hold_scores,evaluation_labels
from ipad_vad.events import anomaly_events,summarize_events
from ipad_vad.learned_detector import sha,write
from evaluate_baseline import metrics
from experiment42_normal import SCENES,RUNS,BRANCHES,OUT,ART,load,cfg,factory,arrays,verify,load_npz

def main():
    if (OUT/'metrics.json').exists():raise RuntimeError('Evaluation complete')
    verify(OUT/'normal_models_checkpoint.json');verify(OUT/'pre_normal_models_protocol.json');verify(OUT/'training_protocol.json');verify(OUT/'testing_observations.json');test=json.loads(Path('results/experiment41/test_data_manifest.json').read_text());manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text());normal=json.loads((OUT/'normal_models_audit.json').read_text());oldmetrics=json.loads(Path('results/experiment41/metrics.json').read_text());oldpred=json.loads(Path('results/experiment41/test_scores_checkpoint.json').read_text());labels_allowed=False
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and '/test_label/' in str(args[0]) and not labels_allowed:raise RuntimeError('Score all branches first')
    sys.addaudithook(guard);files=[Path(__file__),OUT/'normal_models_checkpoint.json',OUT/'testing_observations.json',OUT/'association_gates.json'];write(OUT/'pre_test_scoring_protocol.json',{'labels_opened':False,'file_sha256':{str(p):sha(p) for p in files}});predictions=[];controlcount=0;equalcounts={b:0 for b in BRANCHES};processchecks=0
    for scene in SCENES:
        sp=manifest['subsplits'][scene];process_control={}
        for run in RUNS:
            for branch in BRANCHES:
                m=factory(cfg(run,scene));m.fit([load(branch,run,scene,s) for s in sp['downstream_fit']]);m.calibrate([load(branch,run,scene,s) for s in sp['normal_calibration']]);expected=load_npz(ART/branch/'normal_models'/f'{run}_{scene}_full_model.npz');actual=arrays(m);assert set(expected)==set(actual)
                for k,v in actual.items():np.testing.assert_array_equal(v,expected[k])
                for item in [r for r in test['sequences'] if r['scene']==scene]:
                    seq=item['sequence'];d=load(branch,run,scene,seq,'testing');result=m.score(d);n=int(d['frame_count']);saved={k:hold_scores(d['indices'],v,n) for k,v in result.items() if isinstance(v,np.ndarray)};oldpath=Path('artifacts/experiment41')/run/'predictions'/f'{scene}_{seq}.npz';assert sha(oldpath)==oldpred['file_sha256'][str(oldpath)];old=load_npz(oldpath)
                    # Control predictions are recomputed using historical source observations.
                    if branch=='raw':
                        control=m.score(load('iou',run,scene,seq,'testing'))
                        # The normal stage proves model equivalence before this shortcut.
                        assert next(r for r in normal['full'] if r['branch']==branch and r['run']==run and r['scene']==scene)['exact_model_and_scores_to_41']
                        for k,v in control.items():
                            if isinstance(v,np.ndarray):np.testing.assert_array_equal(hold_scores(d['indices'],v,n),old[k])
                        assert m.threshold==float(old['threshold']);controlcount+=1
                    equal=all(np.array_equal(v,old[k]) for k,v in saved.items()) and m.threshold==float(old['threshold']);equalcounts[branch]+=int(equal)
                    processkeys=[k for k in saved if k not in ['visual','combined']]
                    if run=='A':process_control[(branch,seq)]={k:saved[k] for k in processkeys}
                    else:
                        for k in processkeys:np.testing.assert_array_equal(saved[k],process_control[(branch,seq)][k]);processchecks+=1
                    object_rows=[(role,int(frame),float(score)) for role,frames,scores in result['objects'] for frame,score in zip(frames,scores)];detection_ids=np.concatenate([np.full(len(frames),-1,dtype=int) if role==-1 else np.flatnonzero(d['roles']==role) for role,frames,scores in result['objects']]);saved.update(indices=d['indices'],phases=d['phases'],threshold=np.array(m.threshold),objects=np.array(object_rows),object_detection_indices=detection_ids,boxes=d['boxes'],roles=d['roles'],tracks=d['tracks'],object_frames=d['object_frames']);dest=ART/branch/run/'predictions'/f'{scene}_{seq}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**saved);predictions.append(dest)
                print('Scored',scene,run,branch,flush=True)
    write(OUT/'test_scores_checkpoint.json',{'labels_opened':False,'file_sha256':{str(p):sha(p) for p in predictions},'control_sequences_exact_to_41':controlcount,'new_score_sequences_exact_to_41':equalcounts,'process_array_checks':processchecks});labels_allowed=True;root=Path(json.loads(Path('configs/experiment40_representation.json').read_text())['data_root']);labels={};labelhash={};summaries=[];per=[];events=[];contrasts=[]
    for item in test['sequences']:
        scene,seq=item['scene'],item['sequence'];p=root/scene/'test_label'/f'{int(seq):03}.npy';labels[(scene,seq)]=evaluation_labels(np.load(p),item['source_frames']);labelhash[str(p)]=sha(p)
    for branch in BRANCHES:
        for scene in SCENES:
            for run in RUNS:
                ys=[];ss={k:[] for k in ['visual','process','combined']};ev=[];oldss=[]
                for item in [r for r in test['sequences'] if r['scene']==scene]:
                    seq=item['sequence'];y=labels[(scene,seq)];d=load_npz(ART/branch/run/'predictions'/f'{scene}_{seq}.npz');q=float(d['threshold']);alarm=d['combined']>q;ys.append(y)
                    for k in ss:ss[k].append(d[k])
                    e=[dict(e,branch=branch,run=run,scene=scene,sequence=seq) for e in anomaly_events(y,alarm)];ev+=e;events+=e;oldss.append(load_npz(Path('artifacts/experiment41')/run/'predictions'/f'{scene}_{seq}.npz')['combined']);r={'branch':branch,'run':run,'scene':scene,'sequence':seq,'valid_frames':int((y>=0).sum()),'unknown_frames':int((y<0).sum()),'normal_frames':int((y==0).sum()),'anomaly_frames':int((y==1).sum()),'fp':int((alarm&(y==0)).sum()),'tp':int((alarm&(y==1)).sum()),'q99':q,'events':len(e),'detected_events':sum(x['detected'] for x in e)}
                    for k in ss:mm=metrics(y,d[k]);r[k+'_auroc']=mm['auroc'];r[k+'_ap']=mm['average_precision']
                    per.append(r)
                y=np.concatenate(ys);s={k:np.concatenate(v) for k,v in ss.items()};alarm=s['combined']>q;folds=[r for r in normal['folds'] if r['branch']==branch and r['scene']==scene and r['run']==run];r={'branch':branch,'run':run,'scene':scene,'metrics':{k:metrics(y,v) for k,v in s.items()},'q99':q,'fp':int((alarm&(y==0)).sum()),'tp':int((alarm&(y==1)).sum()),'normal_fpr':float(alarm[y==0].mean()),'anomaly_recall':float(alarm[y==1].mean()),'events':summarize_events(ev),'normal_frames':int((y==0).sum()),'anomaly_frames':int((y==1).sum()),'unknown_frames':int((y<0).sum()),'conditional_normal_holdout_fp':sum(x['frame_alarms'] for x in folds),'conditional_normal_holdout_frames':sum(x['frames'] for x in folds)};summaries.append(r);ref=next(x for x in oldmetrics['variants'] if x['run']==run and x['scene']==scene);old=np.concatenate(oldss)>ref['q99'];contrasts.append({'branch':branch,'run':run,'scene':scene,'combined_auroc_delta':r['metrics']['combined']['auroc']-ref['metrics']['combined']['auroc'],'added_fp':int((alarm&~old&(y==0)).sum()),'removed_fp':int((~alarm&old&(y==0)).sum()),'added_tp':int((alarm&~old&(y==1)).sum()),'removed_tp':int((~alarm&old&(y==1)).sum()),'old_q99':ref['q99'],'new_q99':q})
    macro=[];seed=[]
    for branch in BRANCHES:
        for run in RUNS:
            rr=[r for r in summaries if r['branch']==branch and r['run']==run];macro.append({'branch':branch,'run':run,**{b+'_'+k:float(np.mean([r['metrics'][b][k] for r in rr])) for b in ['visual','combined'] for k in ['auroc','average_precision']},'normal_fpr':float(np.mean([r['normal_fpr'] for r in rr])),'anomaly_recall':float(np.mean([r['anomaly_recall'] for r in rr])),'event_coverage':float(np.mean([r['events']['event_coverage'] for r in rr]))})
        for arm in ['C','D']:
            rr=[r for r in macro if r['branch']==branch and r['run'].startswith(arm)];seed.append({'branch':branch,'arm':arm,'macro':{k:{'mean':float(np.mean([r[k] for r in rr])),'sample_std':float(np.std([r[k] for r in rr],ddof=1))} for k in rr[0] if k not in ['branch','run']}})
    write(OUT/'metrics.json',{'variants':summaries,'scene_macro':macro,'adaptation_seed_summary':seed,'paired_41_contrasts':contrasts,'label_sha256':labelhash,'scope':'Development scenes, not independent confirmation. Association one seed, historical visual C/D three seeds. No ID ground truth. Separate normal q99; conditional q99 holdout does NOT hold out association gate supervision.'});write(OUT/'events.json',events)
    with (OUT/'per_sequence.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(per[0]),lineterminator='\n');w.writeheader();w.writerows(per)
    verify(OUT/'test_scores_checkpoint.json');write(OUT/'evaluation_audit.json',{'normal_models_refitted_exact':64,'predictions_frozen_before_labels':len(predictions),'control_sequences_exact_to_41':controlcount,'new_score_sequences_exact_to_41':equalcounts,'prediction_hashes_unchanged_after_labels':True,'per_sequence_rows':len(per),'metrics_rows':len(summaries)});print(json.dumps(macro,indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
