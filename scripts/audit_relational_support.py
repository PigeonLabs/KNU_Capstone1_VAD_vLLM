"""Inspect normal FIT relation and complete-context support before test extraction."""
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.relational_phase import RelationalPhase
from ipad_vad.context_dwell import complete_context_runs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());scene=cfg['scene'];split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    root=Path(f'artifacts/experiment{cfg["feature_source_experiment"]}/features/{scene}')
    data=[]
    for seq in split['fit']:
        with np.load(root/f'training_{seq}.npz',allow_pickle=False) as f:data.append(dict(f))
    model=RelationalPhase(**cfg['relational_phase'])
    with threadpool_limits(limits=4):model.fit(data)
    rows=[];banks={}
    for seq,d in zip(split['fit'],data):
        phase,valid,chosen,x=model.transform(d);derived=dict(d,phases=phase,relation_valid=valid)
        counts={}
        for (a,b),duration in complete_context_runs(derived):
            key=f'{a}->{b}';banks.setdefault(key,[]).append({'sequence':seq,'duration_frames':duration});counts[key]=counts.get(key,0)+1
        roles=np.unique(d['roles']);coverage={str(int(r)):len(np.unique(d['object_frames'][d['roles']==r]))/len(valid) for r in roles}
        rows.append({'sequence':seq,'samples':len(valid),'valid_relations':int(valid.sum()),'relation_fraction':float(valid.mean()),'role_sample_coverage':coverage,'phase_counts':np.bincount(phase,minlength=model.k).tolist(),'phase_transitions':int(np.sum(phase[1:]!=phase[:-1])),'complete_context_support':counts})
    minimum=cfg['normal_dwell']['minimum_complete_runs']
    contexts={k:{'runs':len(v),'distinct_videos':len(set(r['sequence'] for r in v)),'min_duration':min(r['duration_frames'] for r in v),'max_duration':max(r['duration_frames'] for r in v),'supported':len(v)>=minimum,'observations':v} for k,v in banks.items()}
    n=sum(r['samples'] for r in rows);v=sum(r['valid_relations'] for r in rows)
    result={'scene':scene,'fit_ids':split['fit'],'normal_only':True,'model':model.evidence,'samples':n,'valid_relations':v,'valid_relation_fraction':v/n,'contexts':contexts,'supported_contexts':[k for k,v in contexts.items() if v['supported']],'sequences':rows,'limitations':['Normal observations only; coverage is not object or phase accuracy.','Complete runs within a video are not independent videos.','First/last and missing-observation runs are excluded.']}
    out=Path(f'results/experiment{cfg["experiment"]}');out.mkdir(parents=True,exist_ok=True);(out/'normal_support.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sequences','model','contexts']},indent=2))


if __name__=='__main__':main()
