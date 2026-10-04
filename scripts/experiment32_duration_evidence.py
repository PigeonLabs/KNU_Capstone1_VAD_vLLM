"""Normal-only episode extraction, evidence summaries and immutable model checks."""
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from ipad_vad.dwell_episodes import extract_dwell_episodes

OUT=Path('results/experiment32');ART=Path('artifacts/experiment32')
CFG=json.loads(Path('configs/experiment32.json').read_text());ROOT=Path(CFG['feature_source'])
SPLIT=json.loads(Path(CFG['split_source']).read_text())['R04']
STATUSES=['complete','right_censored','unknown_entry']
REASONS=['phase_change','anchor_changed','target_changed','both_changed','relation_missing','video_end']


def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p,allow_pickle=False) as f:return dict(f)


def protected_paths():
    return [*[ROOT/f'training_{s}.npz' for s in SPLIT['fit']+SPLIT['calibration']],
            *[Path(f'artifacts/experiment31_{gate}/{name}.npz') for gate in ['hold','pool','age'] for name in ['normal_model','normal_calibration_scores']],
            *sorted(Path('artifacts/experiment31/full_normal').glob('*.npz')),
            *sorted(Path('artifacts/experiment31/normal_holdout').glob('*.npz'))]


def install_data_guard():
    allowed={str(p.resolve()) for p in protected_paths()};opened=set()
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]);s=str(path.resolve())
        if any(x in s for x in ['/testing/','/test_label/','/predictions/']) or path.name.startswith('testing_'):
            raise RuntimeError('Experiment32 forbids test data access: '+s)
        if path.suffix in ['.npz','.npy','.jpg','.jpeg','.mp4']:
            if s not in allowed:raise RuntimeError('Data path outside frozen normal allowlist: '+s)
            opened.add(s)
    sys.addaudithook(hook);return opened


def verify_protocol():
    record=json.loads((OUT/'pre_audit_protocol.json').read_text())
    for section in ['code_input_sha256','protected_normal_sha256']:
        for p,h in record[section].items():assert sha(p)==h,p
    return record


def prepare():
    install_data_guard();files=[Path('configs/experiment32.json'),Path('docs/EXPERIMENT32_PLAN.md'),Path(CFG['split_source']),Path(CFG['legacy_identity_audit']),Path('src/ipad_vad/dwell_episodes.py'),Path('src/ipad_vad/transition_evidence.py'),Path('tests/test_dwell_episodes.py'),Path('scripts/experiment32_duration_evidence.py'),Path('scripts/validate_duration_evidence.py')]
    if (OUT/'pre_audit_protocol.json').exists():verify_protocol()
    else:write(OUT/'pre_audit_protocol.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'normal_only':True,'code_input_sha256':{str(p):sha(p) for p in files},'protected_normal_sha256':{str(p):sha(p) for p in protected_paths()}})
    print('Protected normal files:',len(protected_paths()),flush=True)


def lengths(values):
    return None if not values else {'count':len(values),'min':min(values),'median':float(np.median(values)),'max':max(values)}


def summarize(rows):
    videos=sorted({r['sequence'] for r in rows});by_status={}
    for status in STATUSES:
        selected=[r for r in rows if r['status']==status];counts={s:sum(r['sequence']==s for r in selected) for s in videos};nonzero={s:n for s,n in counts.items() if n}
        by_status[status]={'episodes':len(selected),'videos':len(nonzero),'per_video':nonzero,'largest_video_fraction':max(nonzero.values())/len(selected) if selected else None}
    complete=[r['duration_frames'] for r in rows if r['status']=='complete'];censored=[r['censor_lower_bound_frames'] for r in rows if r['status']=='right_censored'];unknown=[r['observation_span_frames'] for r in rows if r['status']=='unknown_entry']
    return {'episodes':len(rows),'videos':len(videos),'status':by_status,'end_reasons':{k:sum(r['end_reason']==k for r in rows) for k in REASONS},'complete_duration_frames':lengths(complete),'censor_lower_bound_frames':lengths(censored),'positive_censor_lower_bound_frames':lengths([v for v in censored if v>0]),'zero_censor_bounds':sum(v==0 for v in censored),'unknown_entry_observation_span_frames':lengths(unknown),'meets_existing_complete_support':len(complete)>=CFG['support_rule_for_reporting_only']['minimum_complete_runs']}


def audit():
    opened=install_data_guard();protocol=verify_protocol();rows=[];sequences=[]
    for part in ['fit','calibration']:
        for seq in SPLIT[part]:
            path=ROOT/f'training_{seq}.npz';d=load(path);result=extract_dwell_episodes(d)
            for row in result['episodes']:rows.append(dict(row,partition=part,sequence=seq))
            assigned=result['sample_episode_ids'];valid=d['relation_valid'];assert np.array_equal(assigned>=0,valid)
            assert sum(r['observed_samples'] for r in result['episodes'])==int(valid.sum())
            sequences.append({'partition':part,'sequence':seq,'source_frames':int(d['frame_count']),'samples':len(assigned),'valid_samples':int(valid.sum()),'missing_samples':int((~valid).sum()),'episodes':len(result['episodes'])})
            write(ART/'sample_assignments'/f'{seq}.json',{k:v.tolist() for k,v in result.items() if k!='episodes'})
    write(OUT/'episodes.json',{'normal_only':True,'episodes':rows})
    summary={}
    for part in ['fit','calibration']:
        selected=[r for r in rows if r['partition']==part];contexts=sorted({f'{r["entry_context"] if r["entry_observed"] else "?"}->{r["phase"]}' for r in selected});context_rows={}
        for key in contexts:context_rows[key]=summarize([r for r in selected if f'{r["entry_context"] if r["entry_observed"] else "?"}->{r["phase"]}'==key])
        summary[part]={'total':summarize(selected),'contexts':context_rows}
    legacy=json.loads(Path(CFG['legacy_identity_audit']).read_text())['runs'];expected=sorted((r['partition'],r['sequence'],r['context'],r['start'],r['end'],r['duration']) for r in legacy if r['identity_continuous_complete']);actual=sorted((r['partition'],r['sequence'],f'{r["entry_context"]}->{r["phase"]}',r['first_observed_source_frame'],r['boundary_source_frame'],float(r['duration_frames'])) for r in rows if r['status']=='complete');assert actual==expected
    verify_protocol();write(OUT/'summary.json',{'normal_only':True,'sequence_counts':{k:len(SPLIT[k]) for k in ['fit','calibration']},'sequences':sequences,'partitions':summary,'legacy_identity_complete_records_exact':True,'protected_normal_files_unchanged':len(protocol['protected_normal_sha256']),'score_or_model_changes':False,'new_anomaly_metrics':None,'note':'Complete durations are sampled-phase endpoint differences. Censor bounds stop at the last real observation. Unknown-entry spans are not duration labels; fit and calibration remain separate.'});write(OUT/'audit_access_log.json',{'guard':'Exact allowlist for numeric/video data and explicit rejection of test/prediction paths. Python open audit hook active throughout extraction and protected-file verification.','opened_data_paths':sorted(opened),'test_data_opened':False});print(json.dumps({p:v['total'] for p,v in summary.items()},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','audit']);a=p.parse_args();globals()[a.stage]()
