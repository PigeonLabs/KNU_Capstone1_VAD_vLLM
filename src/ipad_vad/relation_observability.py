"""Read-only provenance for role candidate filtering and prior-track selection."""
from collections import defaultdict
import numpy as np

REASONS=['no_raw_candidate','no_positive_area','semantic_excluded','area_excluded','no_joint_candidate','candidate_available']


def candidate_masks(area,semantic,upper):
    area=np.asarray(area);semantic=np.asarray(semantic)
    positive=area>0;area_ok=positive&(area<=upper+1e-8);joint=area_ok&semantic
    # Diagnostic precedence is explicit; independent masks retain overlapping failures.
    reason=('no_raw_candidate' if not len(area) else 'no_positive_area' if not positive.any()
            else 'semantic_excluded' if not np.any(positive&semantic) else 'area_excluded' if not area_ok.any()
            else 'no_joint_candidate' if not joint.any() else 'candidate_available')
    return positive,area_ok,joint,reason


def trace_role_candidates(data,anchor_role,target_role,area_upper,raw_margins,gate_margins,margin_threshold=0.):
    n=len(data['indices']);roles=np.asarray(data['roles']);frames=np.asarray(data['object_frames']);tracks=np.asarray(data['tracks']);boxes=np.asarray(data['boxes']);confidence=np.asarray(data['confidence']);raw=np.asarray(raw_margins);gate=np.asarray(gate_margins);m=len(roles)
    if boxes.shape!=(m,4) or any(v.shape!=(m,) for v in [frames,tracks,confidence,raw,gate]):raise ValueError('Aligned candidate metadata required')
    if not all(np.isfinite(v).all() for v in [boxes,confidence,raw,gate]):raise ValueError('Finite candidate metadata required')
    if any(not np.issubdtype(v.dtype,np.integer) for v in [roles,frames,tracks]) or np.any(frames<0) or np.any(frames>=n) or np.any(tracks<0):raise ValueError('Valid integer candidate roles, frames and tracks required')
    prior={'anchor':None,'target':None};rows=[];chosen=np.full((n,2),-1,int)
    for step in range(n):
        record={'sample':step,'source_frame':int(data['indices'][step]),'roles':{}}
        for column,(name,role) in enumerate([('anchor',anchor_role),('target',target_role)]):
            ids=np.flatnonzero((frames==step)&(roles==role));area=np.maximum(boxes[ids,2:]-boxes[ids,:2],0).prod(1);applied=name=='anchor';sem=gate[ids]>margin_threshold if applied else np.ones(len(ids),bool);positive,area_ok,joint,reason=candidate_masks(area,sem,area_upper[role]);eligible=ids[joint];same=eligible[tracks[eligible]==prior[name]];pool=same if len(same) else eligible;index=int(pool[np.argmax(confidence[pool])]) if len(pool) else -1
            previous=prior[name];prior_ids=ids[tracks[ids]==previous];prior_ok=eligible[tracks[eligible]==previous];track=int(tracks[index]) if index>=0 else None
            event='no_selection' if index<0 else 'first_selection' if previous is None else 'retained' if previous==track else 'prior_candidate_absent' if not len(prior_ids) else 'prior_candidate_filtered'
            if event.startswith('prior_candidate'):assert not len(prior_ok)
            candidates=[]
            for j,i in enumerate(ids):
                candidates.append({'detection_index':int(i),'track':int(tracks[i]),'bbox_normalized':boxes[i].tolist(),'confidence':float(confidence[i]),'area':float(area[j]),'positive_area':bool(positive[j]),'area_pass':bool(area_ok[j]),'semantic_pass':bool(sem[j]) if applied else None,'raw_margin':float(raw[i]) if applied else None,'gate_margin':float(gate[i]) if applied else None,'joint_pass':bool(joint[j]),'selected':int(i)==index})
            record['roles'][name]={'role':role,'semantic_applied':applied,'area_upper':area_upper[role],'reason':reason,'raw_count':len(ids),'positive_area_count':int(positive.sum()),'semantic_raw_pass_count':int(sem.sum()) if applied else None,'semantic_positive_pass_count':int(np.sum(sem&positive)) if applied else None,'area_pass_count':int(area_ok.sum()),'joint_pass_count':int(joint.sum()),'prior_track':previous,'prior_raw_candidates':prior_ids.tolist(),'prior_joint_candidates':prior_ok.tolist(),'selected_detection_index':index,'selected_track':track,'selection_event':event,'candidates':candidates}
            chosen[step,column]=index
            if index>=0:prior[name]=track
        record['relation_valid']=bool(np.all(chosen[step]>=0));record['failure_combination']='|'.join(record['roles'][name]['reason'] for name in ['anchor','target']);rows.append(record)
    return rows,chosen,np.all(chosen>=0,axis=1)


def select_boundary_cases(episodes,limit=6):
    groups=defaultdict(list)
    for row in episodes:
        base={'sequence':row['sequence'],'partition':row['partition'],'episode_id':row['episode_id']}
        if row['status']=='right_censored' and row['end_reason']=='relation_missing':groups['censored_missing'].append(dict(base,sample=row['boundary_sample'],source_frame=row['boundary_source_frame']))
        if row['status']=='unknown_entry' and row['start_reason']=='reacquired':groups['unknown_reacquired'].append(dict(base,sample=row['start_sample'],source_frame=row['first_observed_source_frame']))
        if row['status']=='unknown_entry' and row['start_reason'] in ['anchor_changed','target_changed','both_changed']:groups['unknown_track_change'].append(dict(base,sample=row['start_sample'],source_frame=row['first_observed_source_frame']))
        if row['status']=='complete':groups['complete_control'].append(dict(base,sample=row['boundary_sample'],source_frame=row['boundary_source_frame']))
    cases={};memberships={}
    for name in ['censored_missing','unknown_reacquired','unknown_track_change','complete_control']:
        by_video=defaultdict(list)
        for row in sorted(groups[name],key=lambda r:(r['sequence'],r['source_frame'],r['episode_id'])):by_video[row['sequence']].append(row)
        selected=[];level=0
        while len(selected)<limit and any(level<len(v) for v in by_video.values()):
            for seq in sorted(by_video):
                if level<len(by_video[seq]) and len(selected)<limit:selected.append(by_video[seq][level])
            level+=1
        memberships[name]=[]
        for row in selected:
            key=f'{row["sequence"]}_{row["source_frame"]:04d}';memberships[name].append(key)
            if key not in cases:cases[key]={'case_id':key,'sequence':row['sequence'],'partition':row['partition'],'sample':row['sample'],'source_frame':row['source_frame'],'groups':[],'episode_links':[]}
            cases[key]['groups'].append(name);cases[key]['episode_links'].append({'group':name,'episode_id':row['episode_id']})
    return {'selection_rule':'At most six per group, video round-robin in sequence/source-frame order, before visual review. Duplicate boundaries share a case with all memberships.','groups':memberships,'cases':list(cases.values())}
