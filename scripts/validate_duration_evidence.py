"""Independent interval reconstruction and exhaustive normal-video prefix checks."""
import json
from pathlib import Path
import numpy as np
import numpy.testing  # Load validation code before the strict data-access guard.
from ipad_vad.dwell_episodes import extract_dwell_episodes
from experiment32_duration_evidence import (OUT,ART,CFG,ROOT,SPLIT,STATUSES,REASONS,load,write,summarize,verify_protocol,install_data_guard)


def independent(d):
    indices=d['indices'];phase=d['phases'];valid=d['relation_valid'];n=len(indices)
    pairs={i:tuple(int(x) for x in d['tracks'][d['relation_detection_indices'][i]]) for i in np.flatnonzero(valid)}
    groups=[]
    for i in np.flatnonzero(valid):
        i=int(i)
        if not groups or i!=groups[-1][-1]+1 or pairs[i]!=pairs[groups[-1][-1]] or phase[i]!=phase[groups[-1][-1]]:groups.append([i])
        else:groups[-1].append(i)
    def changed(a,b):
        return 'both_changed' if a[0]!=b[0] and a[1]!=b[1] else 'anchor_changed' if a[0]!=b[0] else 'target_changed'
    records=[];assignment=np.full(n,-1,int);age=np.full(n,-1,int);contexts=np.full(n,-1,int)
    for eid,group in enumerate(groups):
        start,last=group[0],group[-1];end=last+1;known=start>0 and valid[start-1] and pairs[start-1]==pairs[start] and phase[start-1]!=phase[start]
        reason='video_start' if start==0 else 'reacquired' if not valid[start-1] else changed(pairs[start-1],pairs[start]) if pairs[start-1]!=pairs[start] else 'phase_change'
        ending='video_end' if end==n else 'relation_missing' if not valid[end] else changed(pairs[last],pairs[end]) if pairs[last]!=pairs[end] else 'phase_change'
        exit_observed=ending=='phase_change';previous=int(phase[start-1]) if known else None
        row={'episode_id':eid,'start_sample':start,'first_observed_source_frame':int(indices[start]),'phase':int(phase[start]),'track_pair':list(pairs[start]),'start_reason':reason,'entry_observed':bool(known),'entry_context':previous,'entry_source_frame':int(indices[start]) if known else None,'end_sample_exclusive':end,'last_observed_sample':last,'last_observed_source_frame':int(indices[last]),'observed_samples':len(group),'observation_span_frames':int(indices[last]-indices[start]),'boundary_sample':end if end<n else None,'boundary_source_frame':int(indices[end]) if end<n else None,'end_reason':ending,'exit_observed':exit_observed,'status':'complete' if known and exit_observed else 'right_censored' if known else 'unknown_entry','duration_frames':int(indices[end]-indices[start]) if known and exit_observed else None,'censor_lower_bound_frames':int(indices[last]-indices[start]) if known and not exit_observed else None}
        records.append(row);assignment[group]=eid
        if known:age[group]=indices[group]-indices[start];contexts[group]=previous
    return {'episodes':records,'sample_episode_ids':assignment,'sample_age_frames':age,'sample_entry_context':contexts,'missing_sample_indices':np.flatnonzero(~valid)}


def prefix(d,n):
    ids=np.flatnonzero(d['object_frames']<n);remap=np.full(len(d['tracks']),-1,int);remap[ids]=np.arange(len(ids));selected=d['relation_detection_indices'][:n].copy();present=selected>=0;selected[present]=remap[selected[present]]
    return dict(phases=d['phases'][:n],indices=d['indices'][:n],relation_valid=d['relation_valid'][:n],relation_detection_indices=selected,tracks=d['tracks'][ids],object_frames=d['object_frames'][ids],frame_count=d['frame_count'])


def main():
    opened=install_data_guard();protocol=verify_protocol();saved=json.loads((OUT/'episodes.json').read_text())['episodes'];summary=json.loads((OUT/'summary.json').read_text());reconstructed=[];samples=prefixes=0
    for part in ['fit','calibration']:
        for seq in SPLIT[part]:
            d=load(ROOT/f'training_{seq}.npz');actual=extract_dwell_episodes(d);expected=independent(d);assert actual['episodes']==expected['episodes'];reconstructed.extend(dict(r,partition=part,sequence=seq) for r in expected['episodes']);local=json.loads((ART/'sample_assignments'/f'{seq}.json').read_text())
            for k in expected:
                if k!='episodes':np.testing.assert_array_equal(actual[k],expected[k]);np.testing.assert_array_equal(expected[k],local[k])
            assert np.array_equal(expected['sample_episode_ids']>=0,d['relation_valid'])
            assert sum(r['observed_samples'] for r in expected['episodes'])+len(expected['missing_sample_indices'])==len(d['indices'])
            for n in range(len(d['indices'])+1):
                p=extract_dwell_episodes(prefix(d,n));prefixes+=1
                for key in ['sample_episode_ids','sample_age_frames','sample_entry_context']:np.testing.assert_array_equal(p[key],expected[key][:n])
                for row in p['episodes']:
                    full=expected['episodes'][row['episode_id']]
                    for key in ['episode_id','start_sample','first_observed_source_frame','phase','track_pair','start_reason','entry_observed','entry_context','entry_source_frame']:assert row[key]==full[key]
                    if row['end_reason']!='video_end':assert row==full
                    else:
                        assert row['end_sample_exclusive']==n and row['last_observed_sample']==n-1
                        assert row['duration_frames'] is None
                        if row['entry_observed']:assert row['censor_lower_bound_frames']==int(d['indices'][n-1])-row['entry_source_frame']
            samples+=len(d['indices'])
    assert saved==reconstructed
    for part in ['fit','calibration']:
        rows=[r for r in reconstructed if r['partition']==part];target=summary['partitions'][part];assert target['total']==summarize(rows)
        assert sum(v['episodes'] for v in target['contexts'].values())==len(rows)
        for context,s in target['contexts'].items():
            selected=[r for r in rows if f'{r["entry_context"] if r["entry_observed"] else "?"}->{r["phase"]}'==context];assert s==summarize(selected)
            assert sum(s['status'][status]['episodes'] for status in STATUSES)==len(selected)
            assert sum(s['end_reasons'][reason] for reason in REASONS)==len(selected)
    legacy=json.loads(Path(CFG['legacy_identity_audit']).read_text())['runs'];old=sorted((r['partition'],r['sequence'],r['context'],r['start'],r['end'],r['duration']) for r in legacy if r['identity_continuous_complete']);new=sorted((r['partition'],r['sequence'],f'{r["entry_context"]}->{r["phase"]}',r['first_observed_source_frame'],r['boundary_source_frame'],float(r['duration_frames'])) for r in reconstructed if r['status']=='complete');assert old==new
    verify_protocol();write(OUT/'validation.json',{'normal_sequences':25,'sample_assignments_verified':samples,'episode_records_independently_reconstructed':len(reconstructed),'causal_prefixes_checked':prefixes,'legacy_complete_records_exact':len(old),'partition_context_status_reason_counts_and_lengths_recomputed':True,'all_valid_samples_assigned_once_and_missing_unassigned':True,'zero_bounds_preserved_no_duration_imputed_for_unknown_entry':True,'future_completion_not_used_in_past_age_or_context':True,'protected_normal_files_unchanged':len(protocol['protected_normal_sha256']),'protocol_hashes_verified':True,'test_data_opened':False,'score_or_model_changes':False});write(OUT/'validation_access_log.json',{'opened_data_paths':sorted(opened),'test_data_opened':False,'guard':'Same exact normal-data allowlist as the extraction run.'});print(json.dumps({'episodes':len(reconstructed),'samples':samples,'prefixes':prefixes,'legacy_complete_records':len(old),'protected_files':len(protocol['protected_normal_sha256'])},indent=2))


if __name__=='__main__':main()
