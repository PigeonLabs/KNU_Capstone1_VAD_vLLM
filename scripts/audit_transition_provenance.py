"""Normal-only transition coverage and local evidence traces; never opens test files."""
import json
from pathlib import Path
from collections import deque
import numpy as np
from ipad_vad.temporal_anchor import TemporalAnchorPhase
from ipad_vad.transition_trace import transition_rows,coverage,select_cases
from audit_missing_age import load,sha


def restore_model(cfg):
    options=json.loads(Path(cfg['relation_config']).read_text());opt=options['anchor_verification'];model=TemporalAnchorPhase(load(opt['text_features'])['text_features'],margin_threshold=opt['margin_threshold'],**options['relational_phase'],**options['temporal_anchor']);saved=load(cfg['relation_model'])
    for key in ['location','scale','centers']:setattr(model,key,saved[key])
    model.area_upper=dict(zip(saved['area_roles'].tolist(),saved['area_upper'].tolist()))
    return model


def frame_trace(data,model,files):
    phase,valid,chosen,desc=model.transform(data)
    for a,b in [(phase,data['phases']),(valid,data['relation_valid']),(chosen,data['relation_detection_indices']),(desc,data['relation_descriptors']),(model.raw_margins(data),data['anchor_margin']),(model.margins(data),data['anchor_gate_margin'])]:np.testing.assert_array_equal(a,b)
    frames=[];history=deque(maxlen=model.smoothing)
    for i,frame in enumerate(data['indices']):
        def detection(j):
            if j<0:return None
            return {'detection_index':int(j),'role':int(data['roles'][j]),'track':int(data['tracks'][j]),'bbox_normalized':data['boxes'][j].tolist(),'confidence':float(data['confidence'][j]),'raw_semantic_margin':float(data['anchor_margin'][j]) if data['roles'][j]==model.anchor_role else None,'temporal_semantic_margin':float(data['anchor_gate_margin'][j]) if data['roles'][j]==model.anchor_role else None}
        raw=None;dist=None
        if valid[i]:
            a,b=data['boxes'][chosen[i]];sa=a[2:]-a[:2];sb=b[2:]-b[:2];ca=(a[:2]+a[2:])/2;cb=(b[:2]+b[2:])/2;aa=np.prod(sa);ab=np.prod(sb);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();raw=np.r_[(cb-ca)/np.maximum(sa,1e-8),np.log(ab/aa),inter/max(aa+ab-inter,1e-8),ca];history.append(raw);np.testing.assert_array_equal(np.mean(history,axis=0),desc[i]);dist=np.sum(((desc[i]-model.location)/model.scale-model.centers)**2,axis=1).tolist()
        else:history.clear()
        assert int(files[int(frame)].stem)==int(frame)
        frames.append({'sample':i,'source_frame':int(frame),'source_path':str(files[int(frame)]),'phase':int(phase[i]),'relation_valid':bool(valid[i]),'selected':{'anchor':detection(chosen[i,0]),'target':detection(chosen[i,1])},'candidates':[detection(j) for j in np.flatnonzero(data['object_frames']==i)],'raw_relation_descriptor':None if raw is None else raw.tolist(),'smoothed_relation_descriptor':desc[i].tolist() if valid[i] else None,'smoothing_observations':len(history),'squared_distances_to_centers':dist})
    return frames


def main():
    cfg=json.loads(Path('configs/experiment29.json').read_text());out=Path('results/experiment29');art=Path('artifacts/experiment29');(art/'traces').mkdir(parents=True,exist_ok=True);protocol=json.loads((out/'pre_audit_protocol.json').read_text())
    for p,h in protocol['protected_sha256'].items():assert sha(p)==h,p
    split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];model=restore_model(cfg);rows=[];sequence_records=[]
    for partition in ['fit','calibration']:
        for seq in split[partition]:
            d=load(Path(cfg['feature_source'])/f'training_{seq}.npz');root=Path(cfg['data_root'])/'R04/training/frames'/seq;files=sorted(root.glob('*.jpg'),key=lambda p:int(p.stem));assert len(files)==int(d['frame_count']) and [int(p.stem) for p in files]==list(range(len(files)));frames=frame_trace(d,model,files);transitions=transition_rows(seq,partition,d,frames);rows.extend(transitions)
            (art/'traces'/f'{seq}.json').write_text(json.dumps({'sequence':seq,'partition':partition,'frames':frames,'transitions':transitions},indent=2)+'\n');sequence_records.append({'sequence':seq,'partition':partition,'source_frames':len(files),'sampled_frames':len(frames),'transitions':len(transitions),'observed_transitions':sum(r['observed_pair'] for r in transitions)})
    result={part:coverage([r for r in rows if r['partition']==part]) for part in ['fit','calibration']};folds=[]
    for held in split['calibration']:
        used=[r for r in rows if r['partition']=='calibration' and r['sequence']!=held];probe=[r for r in rows if r['partition']=='calibration' and r['sequence']==held and r['observed_pair']];c=coverage(used)['observed'];counts=np.array(c['counts']);edges=[]
        for a in range(4):
            for b in range(4):
                selected=[r for r in probe if (r['previous_phase'],r['current_phase'])==(a,b)]
                if selected and counts[a,b]==0:edges.append({'previous':a,'current':b,'held_samples':len(selected),'held_source_frames':[r['source_frame'] for r in selected],'remaining_previous_state_samples':int(counts[a].sum()),'state_supported':bool(counts[a].sum()>=10),'remaining_global_samples':int(counts.sum())})
        folds.append({'held_out':held,'remaining_coverage':c,'held_edges_absent_from_remaining':edges})
    tracks={}
    for part in ['fit','calibration']:
        for name,condition in [('3_to_2',lambda r:r['previous_phase']==3 and r['current_phase']==2),('other_previous_3',lambda r:r['previous_phase']==3 and r['current_phase']!=2)]:
            selected=[r for r in rows if r['partition']==part and r['observed_pair'] and condition(r)];tracks[part+'_'+name]={'transitions':len(selected),'anchor_track_changes':sum(r['anchor_track_changed'] is True for r in selected),'target_track_changes':sum(r['target_track_changed'] is True for r in selected)}
    cases=select_cases(rows,cfg['case_limit_per_group']);(out/'case_selection.json').write_text(json.dumps({'normal_only':True,'selected_before_visual_review':True,'selection_rule':'Two frozen calibration counterexamples, then video-round-robin in sequence/frame order; controls previous=3,current!=2 including self. Max24 per group.','cases':cases},indent=2)+'\n');image_hashes={}
    for case in cases:
        trace=json.loads((art/'traces'/f'{case["sequence"]}.json').read_text());i=case['sample']
        for j in range(i-1,min(i+2,len(trace['frames']))):
            p=trace['frames'][j]['source_path'];image_hashes[p]=sha(p)
    (out/'pre_visual_checkpoint.json').write_text(json.dumps({'selection_sha256':sha(out/'case_selection.json'),'source_normal_image_sha256':image_hashes,'local_trace_sha256':{str(p):sha(p) for p in sorted((art/'traces').glob('*.json'))}},indent=2)+'\n')
    summary={'normal_only':True,'normal_sequences':sequence_records,'coverage':result,'calibration_holdout':folds,'track_changes':tracks,'selected_counts':{g:sum(r['selection_group']==g for r in cases) for g in ['target_3_to_2','control_previous_3']},'all_cached_phase_mask_selection_descriptors_margins_exactly_reconstructed':True,'score_or_model_changes':False,'limitations':['Latent IDs are not semantic action ground truth.','Track changes are detector/tracker metadata, not verified identity errors.','Purposeful visual samples cannot estimate semantic accuracy.','No test inputs/labels or new detector/VLM/CLIP inference.']};(out/'coverage_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'coverage':result,'track_changes':tracks,'selected_counts':summary['selected_counts']},indent=2))


if __name__=='__main__':main()
