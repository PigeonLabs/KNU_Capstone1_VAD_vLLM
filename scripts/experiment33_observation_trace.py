"""Normal candidate-to-episode provenance with frozen visual case selection."""
import argparse,json,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import numpy.testing
from ipad_vad.relation_observability import trace_role_candidates,select_boundary_cases,REASONS
from audit_transition_provenance import restore_model
from experiment32_duration_evidence import protected_paths,load,sha,write

OUT=Path('results/experiment33');ART=Path('artifacts/experiment33');CFG=json.loads(Path('configs/experiment33.json').read_text());ROOT=Path(CFG['feature_source']);SPLIT=json.loads(Path('results/stage00/splits.json').read_text())['R04']


def protected():return protected_paths()+[Path(CFG['relation_model']),Path('artifacts/experiment17/anchor_text_features.npz')]


def guard():
    allowed={str(p.resolve()) for p in protected()};opened=set()
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]);s=str(p.resolve())
        if any(x in s for x in ['/test_label/','/predictions/']) or p.name.startswith('testing_') or ('/testing/' in s and p.suffix not in ['.py','.pyc']):raise RuntimeError('Test data forbidden: '+s)
        if p.suffix.lower() in ['.npz','.npy','.jpg','.jpeg','.png','.mp4']:
            local_image=p.suffix.lower() in ['.jpg','.jpeg','.png'] and p.resolve().is_relative_to((ART/'contact_sheets').resolve())
            if s not in allowed and not local_image:raise RuntimeError('Data path outside normal allowlist: '+s)
            opened.add(s)
    sys.addaudithook(hook);return allowed,opened


def verify_protocol():
    p=json.loads((OUT/'pre_audit_protocol.json').read_text())
    for name in ['code_input_sha256','protected_normal_sha256']:
        for f,h in p[name].items():assert sha(f)==h,f
    return p


def model():
    m=restore_model(CFG);m.reset_on_track_change=True;return m


def prepare():
    guard();files=[Path('configs/experiment33.json'),Path('configs/experiment18.json'),Path(CFG['episode_source']),Path('results/stage00/splits.json'),Path('docs/EXPERIMENT33_PLAN.md'),Path('src/ipad_vad/relation_observability.py'),Path('src/ipad_vad/relational_phase.py'),Path('src/ipad_vad/verified_anchor.py'),Path('src/ipad_vad/temporal_anchor.py'),Path('tests/test_relation_observability.py'),*[Path('scripts')/s for s in ['experiment33_observation_trace.py','validate_observation_trace.py','render_observation_cases.py','audit_transition_provenance.py','experiment32_duration_evidence.py']]]
    if (OUT/'pre_audit_protocol.json').exists():verify_protocol()
    else:write(OUT/'pre_audit_protocol.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'normal_only':True,'code_input_sha256':{str(p):sha(p) for p in files},'protected_normal_sha256':{str(p):sha(p) for p in protected()}})
    print('Normal inputs/model/score files frozen:',len(protected()),flush=True)


def reconstruct(d,m):
    raw=m.raw_margins(d);gate=m.margins(d);p,v,c,x=m.transform(d)
    for a,b in [(raw,d['anchor_margin']),(gate,d['anchor_gate_margin']),(p,d['phases']),(v,d['relation_valid']),(c,d['relation_detection_indices']),(x,d['relation_descriptors'])]:np.testing.assert_array_equal(a,b)
    rows,chosen,valid=trace_role_candidates(d,m.anchor_role,m.target_role,m.area_upper,raw,gate,m.margin_threshold);np.testing.assert_array_equal(chosen,c);np.testing.assert_array_equal(valid,v)
    for row in rows:row['phase']=int(p[row['sample']])
    return rows


def compact(row):
    if row is None:return None
    return {k:row[k] for k in ['sample','source_frame','phase','relation_valid','failure_combination']}|{'roles':{name:{k:v for k,v in r.items() if k not in ['candidates']} for name,r in row['roles'].items()}}


def summarize(rows):
    return {'samples':len(rows),'valid_samples':sum(r['relation_valid'] for r in rows),'role_reasons':{role:{reason:sum(r['roles'][role]['reason']==reason for r in rows) for reason in REASONS} for role in ['anchor','target']},'failure_combinations':dict(sorted(Counter(r['failure_combination'] for r in rows).items())),'selection_events':{role:dict(sorted(Counter(r['roles'][role]['selection_event'] for r in rows).items())) for role in ['anchor','target']}}


def audit():
    allowed,opened=guard();protocol=verify_protocol();m=model();traces={};parts={};sequence_rows=[];gaps=[];pair_changes=[]
    for part in ['fit','calibration']:
        for seq in SPLIT[part]:
            d=load(ROOT/f'training_{seq}.npz');rows=reconstruct(d,m);traces[seq]=rows;parts[seq]=part
            paths=sorted((Path(CFG['data_root'])/'R04/training/frames'/seq).glob('*.jpg'),key=lambda p:int(p.stem));assert len(paths)==int(d['frame_count'])
            for row in rows:
                frame=row['source_frame'];assert int(paths[frame].stem)==frame;row['source_path']=str(paths[frame])
            write(ART/'traces'/f'{seq}.json',{'sequence':seq,'partition':part,'frames':rows});sequence_rows.append({'sequence':seq,'partition':part,**summarize(rows)})
            valid=d['relation_valid'];edges=np.diff(np.r_[False,~valid,False].astype(int))
            for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                before=rows[start-1] if start else None;after=rows[end] if end<len(rows) else None
                pair=lambda r:None if r is None else [r['roles'][name]['selected_track'] for name in ['anchor','target']]
                gaps.append({'sequence':seq,'partition':part,'start_sample':int(start),'end_sample_exclusive':int(end),'missing_samples':int(end-start),'first_missing_source_frame':int(d['indices'][start]),'last_missing_source_frame':int(d['indices'][end-1]),'preceding_pair':pair(before),'reacquired_pair':pair(after),'reacquired_source_frame':None if after is None else after['source_frame'],'complete_gap':before is not None and after is not None,'same_pair_reacquired':pair(before)==pair(after) if before is not None and after is not None else None,'first_missing_to_reacquisition_frames':None if after is None else int(d['indices'][end]-d['indices'][start]),'failure_combinations':dict(Counter(r['failure_combination'] for r in rows[start:end]))})
            for i in range(1,len(rows)):
                if not(valid[i-1] and valid[i]):continue
                changes=[name for name in ['anchor','target'] if rows[i-1]['roles'][name]['selected_track']!=rows[i]['roles'][name]['selected_track']]
                if changes:pair_changes.append({'sequence':seq,'partition':part,'changed_roles':changes,'previous':compact(rows[i-1]),'current':compact(rows[i])})
    episodes=json.loads(Path(CFG['episode_source']).read_text())['episodes'];links=[]
    for ep in episodes:
        seq=ep['sequence'];rows=traces[seq];start=ep['start_sample'];end=ep['end_sample_exclusive'];links.append({'partition':ep['partition'],'sequence':seq,'episode_id':ep['episode_id'],'status':ep['status'],'start_reason':ep['start_reason'],'end_reason':ep['end_reason'],'before_start':compact(rows[start-1]) if start else None,'at_start':compact(rows[start]),'last_observed':compact(rows[end-1]),'at_end_boundary':compact(rows[end]) if end<len(rows) else None})
    summary={part:summarize([r for seq,rows in traces.items() if parts[seq]==part for r in rows]) for part in ['fit','calibration']}
    write(OUT/'candidate_audit.json',{'normal_only':True,'partitions':summary,'sequences':sequence_rows,'all_cached_phases_descriptors_selection_valid_and_margins_reproduced':True,'score_or_model_changes':False,'diagnostic_reason_order':REASONS,'raw_candidate_definition':'Cached post-detector-threshold/NMS/role candidates, before relation-only semantic/area gates. Not all raw detector proposals.'});write(OUT/'gap_and_track_audit.json',{'gaps':gaps,'consecutive_observed_pair_changes':pair_changes});write(OUT/'episode_boundary_trace.json',{'episodes':links})
    selection=select_boundary_cases(episodes,CFG['case_limit_per_group']);write(OUT/'case_selection.json',selection);image_hashes={};contexts=[]
    for case in selection['cases']:
        rows=traces[case['sequence']];i=case['sample'];frames=[rows[j] for j in [i-1,i,i+1] if 0<=j<len(rows)];contexts.append({'case':case,'frames':frames})
        for frame in frames:
            p=frame['source_path'];allowed.add(str(Path(p).resolve()));image_hashes[p]=sha(p)
    write(OUT/'case_context.json',{'normal_only':True,'posthoc_next_frame_not_inference_input':True,'cases':contexts});write(OUT/'pre_visual_checkpoint.json',{'selection_sha256':sha(OUT/'case_selection.json'),'case_context_sha256':sha(OUT/'case_context.json'),'source_normal_image_sha256':image_hashes,'local_trace_sha256':{str(ART/'traces'/f'{s}.json'):sha(ART/'traces'/f'{s}.json') for s in traces}})
    verify_protocol();write(OUT/'audit_access_log.json',{'opened_data_paths':sorted(opened),'test_data_opened':False,'normal_only':True});print(json.dumps({'summary':summary,'cases':selection['groups'],'unique_cases':len(selection['cases'])},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','audit']);a=p.parse_args();globals()[a.stage]()
