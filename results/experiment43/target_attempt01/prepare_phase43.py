"""Normal-only teacher target audit before any phase-head optimization."""
import copy,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image,ImageDraw
from threadpoolctl import threadpool_limits
from ipad_vad.learned_detector import sha,write
from ipad_vad.data import frames_in_order
from experiment41_phases import phase_models,load_npz
OUT=Path('results/experiment43');ART=Path('artifacts/experiment43')

def main():
    if (OUT/'target_audit.json').exists():raise RuntimeError('Preserve target audit')
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            s=str(args[0])
            if '/test_label/' in s or '/testing/' in s or Path(s).name.startswith('testing_'):raise RuntimeError('Normal only')
    sys.addaudithook(guard);OUT.mkdir(exist_ok=True);ART.mkdir(exist_ok=True);split=json.loads(Path('results/experiment40/data_manifest.json').read_text())['subsplits'];source=json.loads(Path('results/experiment41/detector_training_extraction.json').read_text());hashes={(r['scene'],r['sequence']):r['sha256'] for r in source['sequences']};models=phase_models();records=[];sourcehash={};centers={};evidence={};review=[];root=Path(json.loads(Path('configs/experiment40_representation.json').read_text())['data_root'])
    for scene in ['R01','R02','R03','R04']:
        cache={};observations={};k=3 if scene=='R01' else 4
        # Only train/validation are touched. Calibration remains unopened here.
        for group in ['representation_train','representation_validation']:
            for seq in split[scene][group]:
                p=Path('artifacts/experiment41/detections/features')/scene/f'training_{seq}.npz';assert sha(p)==hashes[(scene,seq)];sourcehash[str(p)]=sha(p);d=load_npz(p);cache[seq]=d
                if scene=='R01':
                    old,valid,chosen,positions=models[scene].transform(d);history=[];xx=[]
                    for pos in positions:
                        if np.isfinite(pos):history.append(pos);history=history[-3:]
                        xx.append([np.mean(history) if history else 0.])
                    x=np.array(xx)
                elif scene=='R02':valid=np.ones(len(d['indices']),bool);chosen=np.full(len(valid),-1);x=d['phase_similarity'];old=d['phases']
                else:
                    old,valid,chosen,desc=models[scene].transform(d);x=(desc-models[scene].location)/models[scene].scale
                    if scene=='R04':x=np.arcsinh(x)
                np.testing.assert_array_equal(old,d['phases']);observations[seq]=(valid,chosen,x)
        if scene!='R02':
            x=np.concatenate([observations[s][2][observations[s][0]] for s in split[scene]['representation_train']]);times=np.concatenate([cache[s]['indices'][observations[s][0]]/max(int(cache[s]['frame_count'])-1,1) for s in split[scene]['representation_train']]);km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(x)
            if scene=='R01':order=np.argsort(km.cluster_centers_[:,0])
            else:order=np.argsort([np.median(times[km.labels_==i]) for i in range(k)],kind='stable')
            centers[scene]=km.cluster_centers_[order];evidence[scene]={'fit_train_only':True,'centers':centers[scene].tolist(),'cluster_fit_counts':[int((km.labels_==i).sum()) for i in order],'median_train_relative_position':[float(np.median(times[km.labels_==i])) for i in order],'ordering':'Increasing x for R01; normal-train median relative frame position permutes R03/R04 IDs only, never inference input.','selection_gates_scaler_unchanged':True}
        else:evidence[scene]={'text_teacher_unchanged':True,'target':'Existing frozen CLIP description similarities, raw winner; confidence gap >=0.01 and agreement with historical causal-smoothed phase required.'}
        for group in ['representation_train','representation_validation']:
            for seq in split[scene][group]:
                d=cache[seq];valid,chosen,x=observations[seq]
                if scene=='R02':
                    order=np.argsort(-x,axis=1);winner=order[:,0];margin=x[np.arange(len(x)),order[:,0]]-x[np.arange(len(x)),order[:,1]];keep=(margin>=.01)&(winner==d['phases']);dist=-x[np.arange(len(x)),winner]
                else:
                    distance=((x[:,None,:]-centers[scene][None,:,:])**2).sum(-1);order=np.argsort(distance,axis=1);winner=order[:,0];d1=distance[np.arange(len(x)),order[:,0]];d2=distance[np.arange(len(x)),order[:,1]];margin=(d2-d1)/np.maximum(d2,1e-9);keep=valid&(margin>=.1);dist=d1
                teacher=[];last=0
                for w,v in zip(winner,valid):
                    if v:last=int(w)
                    teacher.append(last)
                teacher=np.array(teacher);dest=ART/'targets'/scene/f'training_{seq}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,labels=winner,teacher_phases=teacher,valid=valid,keep=keep,margin=margin,indices=d['indices'],chosen=chosen)
                records.append({'scene':scene,'sequence':seq,'partition':group,'samples':len(valid),'valid':int(valid.sum()),'selected':int(keep.sum()),'selected_state_counts':np.bincount(winner[keep],minlength=k).tolist(),'teacher_phase_counts':np.bincount(teacher,minlength=k).tolist(),'old_phase_counts':np.bincount(d['phases'],minlength=k).tolist(),'target_sha256':sha(dest)})
                for state in range(k):
                    for t in np.flatnonzero(keep&(winner==state)):
                        review.append({'scene':scene,'sequence':seq,'partition':group,'state':state,'sample':int(t),'source_frame':int(d['indices'][t]),'distance':float(dist[t]),'margin':float(margin[t])})
        print(scene,evidence[scene],flush=True)
    np.savez_compressed(ART/'teacher_centers.npz',**centers);write(OUT/'teacher_fit.json',evidence);selected=[]
    for scene in ['R01','R02','R03','R04']:
        for part in ['representation_train','representation_validation']:
            for state in range(3 if scene=='R01' else 4):
                rr=[r for r in review if r['scene']==scene and r['partition']==part and r['state']==state]
                # One nearest prototype and one lowest accepted margin in another video.
                if not rr:continue
                first=min(rr,key=lambda r:(r['distance'],r['sequence'],r['sample']));selected.append(dict(first,selection='nearest_teacher_prototype'))
                other=[r for r in rr if r['sequence']!=first['sequence']]
                if other:selected.append(dict(min(other,key=lambda r:(r['margin'],r['sequence'],r['sample'])),selection='lowest_accepted_margin_different_video'))
    dest=ART/'target_review';dest.mkdir(exist_ok=True)
    for offset in range(0,len(selected),12):
        board=Image.new('RGB',(1536,1152),'#151515');draw=ImageDraw.Draw(board)
        for cell,r in enumerate(selected[offset:offset+12]):
            x=(cell%4)*384;y=(cell//4)*384;images=frames_in_order(root/r['scene']/'training/frames'/r['sequence']);p=images[r['source_frame']];im=Image.open(p).convert('RGB').resize((384,352));board.paste(im,(x,y+32));draw.text((x+3,y+3),f'{offset+cell}: {r["scene"]}/{r["sequence"]} s{r["state"]} f{r["source_frame"]} {r["partition"].removeprefix("representation_")}',fill='white');r['image_sha256']=sha(p);r['image_path']=str(p.relative_to(root));r['board']=f'targets_{offset//12:02}.jpg';r['status']='unreviewed'
        board.save(dest/f'targets_{offset//12:02}.jpg',quality=94)
    write(OUT/'target_review_drafts.json',selected);write(OUT/'target_audit.json',{'normal_only':True,'teacher_fit_only_train70':True,'calibration_or_test_used':False,'source_sha256':sourcehash,'teacher_centers_sha256':sha(ART/'teacher_centers.npz'),'sequences':records,'review_frames':len(selected),'rules':{'R01_R03_R04_relative_distance_gap':.1,'R02_text_cosine_gap':.01},'interpretation':'Weak visual/spatial or latent relation targets, never human action GT. Teacher refit control required to separate target adaptation from learned classifier effects.'});print('Review frames',len(selected))
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
