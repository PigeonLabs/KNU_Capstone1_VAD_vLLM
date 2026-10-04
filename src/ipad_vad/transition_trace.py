"""Read-only provenance for latent transitions; no semantic labels or scoring changes."""
from collections import defaultdict, deque
import numpy as np


def transition_rows(sequence, partition, data, frames):
    phases=np.asarray(data['phases']);valid=np.asarray(data['relation_valid']);indices=np.asarray(data['indices'])
    if phases.ndim!=1 or valid.dtype!=np.bool_ or valid.shape!=phases.shape or indices.shape!=phases.shape:
        raise ValueError('Aligned one-dimensional phases, indices and boolean mask required')
    if len(frames)!=len(phases):raise ValueError('Frame trace must align with sampled data')
    rows=[]
    for i in range(1,len(phases)):
        before,after=frames[i-1],frames[i]
        changes={}
        for role in ['anchor','target']:
            a,b=before['selected'][role],after['selected'][role]
            changes[role+'_track_changed']=None if a is None or b is None else a['track']!=b['track']
        rows.append({'case_id':f'{partition}_{sequence}_{int(indices[i]):06d}','sequence':sequence,'partition':partition,'sample':i,'source_frame':int(indices[i]),'previous_source_frame':int(indices[i-1]),'previous_phase':int(phases[i-1]),'current_phase':int(phases[i]),'observed_pair':bool(valid[i-1] and valid[i]),**changes})
    return rows


def coverage(rows,k=4):
    result={}
    for name in ['all','observed']:
        matrix=np.zeros((k,k),int);videos=defaultdict(set);by_video={}
        for r in rows:
            if name=='observed' and not r['observed_pair']:continue
            a,b=r['previous_phase'],r['current_phase'];matrix[a,b]+=1;videos[a,b].add(r['sequence'])
            if r['sequence'] not in by_video:by_video[r['sequence']]=np.zeros((k,k),int)
            by_video[r['sequence']][a,b]+=1
        result[name]={'counts':matrix.tolist(),'distinct_videos':[[len(videos[a,b]) for b in range(k)] for a in range(k)],'by_video':{s:m.tolist() for s,m in sorted(by_video.items())}}
    return result


def select_cases(rows,limit=24):
    if not isinstance(limit,int) or isinstance(limit,bool) or limit<2:raise ValueError('Case budget must include two mandatory failures')
    mandatory=[r for r in rows if r['sequence']=='02' and r['source_frame'] in [152,376] and r['partition']=='calibration']
    if len(mandatory)!=2 or any(not r['observed_pair'] or (r['previous_phase'],r['current_phase'])!=(3,2) for r in mandatory):
        raise ValueError('Frozen normal counterexamples do not match source trace')
    result=[]
    for group in ['target_3_to_2','control_previous_3']:
        chosen=list(sorted(mandatory,key=lambda r:r['source_frame'])) if group=='target_3_to_2' else []
        chosen_ids={r['case_id'] for r in chosen};queues=defaultdict(deque)
        for r in sorted(rows,key=lambda r:(r['sequence'],r['source_frame'])):
            match=r['observed_pair'] and r['previous_phase']==3 and ((r['current_phase']==2)==(group=='target_3_to_2'))
            if match and r['case_id'] not in chosen_ids:queues[r['sequence']].append(r)
        while len(chosen)<limit and any(queues.values()):
            for seq in sorted(queues):
                if queues[seq] and len(chosen)<limit:chosen.append(queues[seq].popleft())
        for r in chosen:result.append(dict(r,selection_group=group,selection_reason='frozen_normal_counterexample' if r in mandatory else 'video_round_robin'))
    return result
