"""Independently reconstruct normal filtering, selections, boundaries and review evidence."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import numpy.testing
from experiment33_observation_trace import OUT,ART,CFG,ROOT,SPLIT,guard,verify_protocol,model,load,write,sha,summarize,compact


def independently_check(d,m,rows):
    features=np.asarray(d['crop_features'],float);text=load('artifacts/experiment17/anchor_text_features.npz')['text_features'].astype(float);text/=np.linalg.norm(text,axis=1,keepdims=True);raw=(features/np.linalg.norm(features,axis=1,keepdims=True))@(text[0]-text[1]);np.testing.assert_array_equal(raw,d['anchor_margin']);gate=raw.copy();hist={};last={}
    for i in np.flatnonzero(d['roles']==m.anchor_role)[np.argsort(d['object_frames'][d['roles']==m.anchor_role],kind='stable')]:
        track=int(d['tracks'][i]);step=int(d['object_frames'][i]);history=hist.get(track,[]) if last.get(track)==step-1 else [];hist[track]=(history+[raw[i]])[-3:];last[track]=step;gate[i]=np.median(hist[track])
    np.testing.assert_array_equal(gate,d['anchor_gate_margin']);previous={'anchor':None,'target':None};chosen=np.full((len(rows),2),-1,int)
    for step,row in enumerate(rows):
        assert row['sample']==step and row['source_frame']==int(d['indices'][step]) and row['phase']==int(d['phases'][step])
        for column,(name,role) in enumerate([('anchor',m.anchor_role),('target',m.target_role)]):
            ids=[i for i in range(len(d['roles'])) if d['object_frames'][i]==step and d['roles'][i]==role];areas={i:float(np.maximum(d['boxes'][i,2:]-d['boxes'][i,:2],0).prod()) for i in ids};positive=[i for i in ids if areas[i]>0];semantic=[i for i in ids if name=='target' or gate[i]>m.margin_threshold];area_ok=[i for i in positive if areas[i]<=m.area_upper[role]+1e-8];joint=[i for i in area_ok if i in semantic]
            reason='candidate_available'
            if not ids:reason='no_raw_candidate'
            elif not positive:reason='no_positive_area'
            elif not set(positive)&set(semantic):reason='semantic_excluded'
            elif not area_ok:reason='area_excluded'
            elif not joint:reason='no_joint_candidate'
            prior=[i for i in ids if int(d['tracks'][i])==previous[name]];prior_ok=[i for i in joint if i in prior];pool=prior_ok or joint;index=max(pool,key=lambda i:float(d['confidence'][i])) if pool else -1;track=int(d['tracks'][index]) if index>=0 else None;event='no_selection' if index<0 else 'first_selection' if previous[name] is None else 'retained' if track==previous[name] else 'prior_candidate_absent' if not prior else 'prior_candidate_filtered';r=row['roles'][name]
            expected={'role':role,'semantic_applied':name=='anchor','area_upper':m.area_upper[role],'reason':reason,'raw_count':len(ids),'positive_area_count':len(positive),'semantic_raw_pass_count':len(semantic) if name=='anchor' else None,'semantic_positive_pass_count':len(set(positive)&set(semantic)) if name=='anchor' else None,'area_pass_count':len(area_ok),'joint_pass_count':len(joint),'prior_track':previous[name],'prior_raw_candidates':prior,'prior_joint_candidates':prior_ok,'selected_detection_index':index,'selected_track':track,'selection_event':event}
            assert {k:r[k] for k in expected}==expected
            assert [x['detection_index'] for x in r['candidates']]==ids
            for candidate in r['candidates']:
                i=candidate['detection_index'];assert candidate['area']==areas[i] and candidate['track']==int(d['tracks'][i]);np.testing.assert_array_equal(candidate['bbox_normalized'],d['boxes'][i]);assert candidate['confidence']==float(d['confidence'][i]);assert candidate['positive_area']==(i in positive) and candidate['area_pass']==(i in area_ok) and candidate['joint_pass']==(i in joint) and candidate['selected']==(i==index)
                assert candidate['semantic_pass']==((i in semantic) if name=='anchor' else None)
                assert candidate['raw_margin']==(float(raw[i]) if name=='anchor' else None) and candidate['gate_margin']==(float(gate[i]) if name=='anchor' else None)
            chosen[step,column]=index
            if index>=0:previous[name]=track
        assert row['relation_valid']==bool(np.all(chosen[step]>=0));assert row['failure_combination']=='|'.join(row['roles'][name]['reason'] for name in ['anchor','target'])
    np.testing.assert_array_equal(chosen,d['relation_detection_indices']);np.testing.assert_array_equal(np.all(chosen>=0,axis=1),d['relation_valid']);p,v,c,x=m.transform(d)
    for a,b in [(p,d['phases']),(v,d['relation_valid']),(c,chosen),(x,d['relation_descriptors'])]:np.testing.assert_array_equal(a,b)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--before-review',action='store_true');args=parser.parse_args();allowed,opened=guard();protocol=verify_protocol();m=model();traces={};parts={};counts=0;check=json.loads((OUT/'pre_visual_checkpoint.json').read_text());audit=json.loads((OUT/'candidate_audit.json').read_text());links=json.loads((OUT/'episode_boundary_trace.json').read_text())['episodes'];episodes=json.loads(Path(CFG['episode_source']).read_text())['episodes']
    for part in ['fit','calibration']:
        for seq in SPLIT[part]:
            d=load(ROOT/f'training_{seq}.npz');path=ART/'traces'/f'{seq}.json';assert sha(path)==check['local_trace_sha256'][str(path)];rows=json.loads(path.read_text())['frames'];independently_check(d,m,rows);traces[seq]=rows;parts[seq]=part;counts+=len(rows);record=next(x for x in audit['sequences'] if x['sequence']==seq);assert {k:record[k] for k in summarize(rows)}==summarize(rows)
    for part in ['fit','calibration']:assert audit['partitions'][part]==summarize([r for seq,rows in traces.items() if parts[seq]==part for r in rows])
    for ep,link in zip(episodes,links):
        assert (ep['partition'],ep['sequence'],ep['episode_id'])==(link['partition'],link['sequence'],link['episode_id']);rows=traces[ep['sequence']];s=ep['start_sample'];e=ep['end_sample_exclusive']
        for key,index in [('before_start',s-1),('at_start',s),('last_observed',e-1),('at_end_boundary',e)]:assert link[key]==(compact(rows[index]) if 0<=index<len(rows) else None)
    gap_audit=json.loads((OUT/'gap_and_track_audit.json').read_text());expected_gaps=[];expected_changes=[]
    for seq,rows in traces.items():
        i=0
        while i<len(rows):
            if rows[i]['relation_valid']:i+=1;continue
            start=i
            while i<len(rows) and not rows[i]['relation_valid']:i+=1
            expected_gaps.append((seq,start,i))
        for i in range(1,len(rows)):
            if rows[i-1]['relation_valid'] and rows[i]['relation_valid']:
                changed=[name for name in ['anchor','target'] if rows[i-1]['roles'][name]['selected_track']!=rows[i]['roles'][name]['selected_track']]
                if changed:expected_changes.append((seq,i,changed))
    assert [(g['sequence'],g['start_sample'],g['end_sample_exclusive']) for g in gap_audit['gaps']]==expected_gaps
    assert [(g['sequence'],g['current']['sample'],g['changed_roles']) for g in gap_audit['consecutive_observed_pair_changes']]==expected_changes
    for gap in gap_audit['gaps']:
        rows=traces[gap['sequence']];s=gap['start_sample'];e=gap['end_sample_exclusive'];pair=lambda i:[rows[i]['roles'][name]['selected_track'] for name in ['anchor','target']]
        assert gap['missing_samples']==e-s and gap['first_missing_source_frame']==rows[s]['source_frame'] and gap['last_missing_source_frame']==rows[e-1]['source_frame'];assert gap['preceding_pair']==(pair(s-1) if s else None);assert gap['reacquired_pair']==(pair(e) if e<len(rows) else None);assert gap['same_pair_reacquired']==(pair(s-1)==pair(e) if s and e<len(rows) else None);assert gap['failure_combinations']==dict(Counter(r['failure_combination'] for r in rows[s:e]));assert gap['first_missing_to_reacquisition_frames']==(rows[e]['source_frame']-rows[s]['source_frame'] if e<len(rows) else None)
    selection=json.loads((OUT/'case_selection.json').read_text());assert sha(OUT/'case_selection.json')==check['selection_sha256'];assert sha(OUT/'case_context.json')==check['case_context_sha256']
    for group in selection['groups']:
        candidates=[]
        for ep in episodes:
            match=(group=='censored_missing' and ep['status']=='right_censored' and ep['end_reason']=='relation_missing') or (group=='unknown_reacquired' and ep['status']=='unknown_entry' and ep['start_reason']=='reacquired') or (group=='unknown_track_change' and ep['status']=='unknown_entry' and ep['start_reason'] in ['anchor_changed','target_changed','both_changed']) or (group=='complete_control' and ep['status']=='complete')
            if match:
                frame=ep['boundary_source_frame'] if group in ['censored_missing','complete_control'] else ep['first_observed_source_frame'];candidates.append((ep['sequence'],frame,ep['episode_id']))
        ranked=[]
        for seq in sorted({r[0] for r in candidates}):
            for ordinal,row in enumerate(sorted(r for r in candidates if r[0]==seq)):ranked.append((ordinal,*row))
        expected=[f'{seq}_{frame:04d}' for _,seq,frame,_ in sorted(ranked)[:CFG['case_limit_per_group']]];assert selection['groups'][group]==expected
    contexts=json.loads((OUT/'case_context.json').read_text())['cases']
    for context in contexts:
        c=context['case'];rows=traces[c['sequence']];i=c['sample'];assert context['frames']==[rows[j] for j in [i-1,i,i+1] if 0<=j<len(rows)]
    for p,h in check['source_normal_image_sha256'].items():allowed.add(str(Path(p).resolve()));assert sha(p)==h
    pages=0
    if not args.before_review:
        manifest=json.loads((OUT/'local_contact_sheet_manifest.json').read_text());assert manifest['renderer_sha256']==sha('scripts/render_observation_cases.py');seen=[]
        for page in manifest['pages']:assert sha(page['path'])==page['sha256'];seen+=page['case_ids'];pages+=1
        assert seen==[c['case_id'] for c in selection['cases']]
        review=json.loads((OUT/'visual_review.json').read_text());assert sorted(x['case_id'] for x in review['cases'])==sorted(seen);assert all(x['semantic_ground_truth'] is None and x['action_boundary_ground_truth'] is None and x['observation'] and x['uncertainty'] for x in review['cases'])
    verify_protocol();write(OUT/('pre_visual_validation.json' if args.before_review else 'validation.json'),{'normal_sequences':25,'samples_independently_checked':counts,'episode_boundaries_checked':len(links),'missing_gaps_checked':len(expected_gaps),'consecutive_pair_changes_checked':len(expected_changes),'selected_unique_cases':len(selection['cases']),'source_images_hashed':len(check['source_normal_image_sha256']),'reviewed_contact_sheets':pages,'raw_temporal_margins_selection_phase_descriptors_exact':True,'filter_counts_and_disjoint_reason_totals_reconstructed':True,'protected_normal_files_unchanged':len(protocol['protected_normal_sha256']),'score_or_model_changes':False,'test_data_opened':False});write(OUT/('pre_visual_validation_access.json' if args.before_review else 'validation_access.json'),{'opened_data_paths':sorted(opened),'test_data_opened':False});print('Validated normal samples:',counts,'cases:',len(selection['cases']),'review pages:',pages,flush=True)


if __name__=='__main__':main()
