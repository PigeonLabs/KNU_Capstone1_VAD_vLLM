"""Independent metric/count and saved-artifact integrity validation."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import rankdata
from ipad_vad.data import evaluation_labels
from ipad_vad.learned_detector import sha,write
from experiment42_normal import OUT,ART,SCENES,RUNS,BRANCHES,verify,load_npz

def independent(y,s):
    keep=y>=0;y=y[keep];s=s[keep];n1=int(y.sum());n0=len(y)-n1
    auc=float((rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n0*n1)) if n0 and n1 else None
    if not n1:return auc,None
    order=np.argsort(-s,kind='stable');yy=y[order];ss=s[order];ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1];tp=np.cumsum(yy)[ends];precision=tp/(ends+1);recall=tp/n1;ap=float(np.sum(np.diff(np.r_[0,recall])*precision));return auc,ap

def main():
    protocols=['training_protocol','training_observations','pre_normal_models_protocol','normal_models_checkpoint','testing_observations','pre_test_scoring_protocol','test_scores_checkpoint']
    for name in protocols:verify(OUT/f'{name}.json')
    tr=json.loads((OUT/'training.json').read_text());assert sha(tr['checkpoint'])==tr['checkpoint_sha256'];assert sha(OUT/'training_protocol.json')==tr['training_protocol_sha256'];assert sha(OUT/'history.json')==tr['history_sha256'];gates=json.loads((OUT/'association_gates.json').read_text())
    for p,h in gates['source_sha256'].items():assert sha(p)==h
    metric=json.loads((OUT/'metrics.json').read_text());test=json.loads(Path('results/experiment41/test_data_manifest.json').read_text());root=Path(json.loads(Path('configs/experiment40_representation.json').read_text())['data_root']);labels={};checks=0;objectchecks=0
    for r in test['sequences']:
        p=root/r['scene']/'test_label'/f'{int(r["sequence"]):03}.npy';assert sha(p)==metric['label_sha256'][str(p)];labels[(r['scene'],r['sequence'])]=evaluation_labels(np.load(p),r['source_frames'])
    assert sum(int((y>=0).sum()) for y in labels.values())==31550;assert sum(int((y<0).sum()) for y in labels.values())==1912
    for r in metric['variants']:
        ds=[];ys=[]
        for item in [x for x in test['sequences'] if x['scene']==r['scene']]:
            d=load_npz(ART/r['branch']/r['run']/'predictions'/f'{r["scene"]}_{item["sequence"]}.npz');ys.append(labels[(r['scene'],item['sequence'])]);ds.append(d);obj=d['objects'];idx=d['object_detection_indices'];assert len(obj)==len(idx)
            for (role,frame,score),di in zip(obj,idx):
                if role==-1:assert di==-1
                else:assert d['roles'][di]==role and d['object_frames'][di]==frame
            objectchecks+=1;assert float(d['threshold'])==r['q99'];assert np.isfinite(d['combined']).all()
        y=np.concatenate(ys)
        for k in ['visual','process','combined']:
            score=np.concatenate([d[k] for d in ds]);auc,ap=independent(y,score);np.testing.assert_allclose(auc,r['metrics'][k]['auroc'],rtol=0,atol=1e-12);np.testing.assert_allclose(ap,r['metrics'][k]['average_precision'],rtol=0,atol=1e-12);checks+=2
        alarm=np.concatenate([d['combined']>r['q99'] for d in ds]);assert int((alarm&(y==0)).sum())==r['fp'];assert int((alarm&(y==1)).sum())==r['tp'];assert float(alarm[y==0].mean())==r['normal_fpr'];assert float(alarm[y==1].mean())==r['anomaly_recall']
        total=detected=0
        for y,d in zip(ys,ds):
            anomalous=y==1;changes=np.diff(np.r_[False,anomalous,False].astype(int));starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1);total+=len(starts);detected+=sum(bool((d['combined'][a:b]>r['q99']).any()) for a,b in zip(starts,ends))
        assert total==r['events']['events'];assert detected==r['events']['detected_events']
    # Check all unchanged detection metadata and exact cached visual features through overlays.
    for split in ['training','testing']:
        record=json.loads((OUT/f'{split}_observations.json').read_text())
        for r in record['sequences']:
            new=load_npz(ART/r['branch']/'observations'/r['scene']/f'{split}_{r["sequence"]}.npz');old=load_npz(Path('artifacts/experiment41/detections/features')/r['scene']/f'{split}_{r["sequence"]}.npz')
            for k in ['indices','frame_count','object_frames','roles','boxes','confidence','phase_similarity']:np.testing.assert_array_equal(new[k],old[k])
            for t in range(len(new['indices'])):
                ix=new['object_frames']==t;assert len(set(new['tracks'][ix].tolist()))==int(ix.sum())
    write(OUT/'validation.json',{'status':'passed','protocol_hash_sets_verified':len(protocols),'independent_auc_ap_checks':checks,'prediction_object_alignment_checks':objectchecks,'valid_frames':31550,'unknown_frames':1912,'all_events_and_operating_counts_independently_verified':True,'no_duplicate_track_per_sample':True,'train_test_detector_boxes_and_confidence_unchanged':True});print('Validation passed',checks,objectchecks)
if __name__=='__main__':main()
