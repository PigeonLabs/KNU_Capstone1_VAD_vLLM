"""Trace extra normal holdout alarms to empirical route-reference maxima."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process
from evaluate_route_holdout import load_cache


def load(p):
    with np.load(p,allow_pickle=False) as f:return dict(f)


def pct(ref,value):
    ref=np.sort(ref);return float((np.searchsorted(ref,value,'left')+np.searchsorted(ref,value,'right'))/(2*len(ref)))


def main():
    out=Path('results/experiment20');art=Path('artifacts/experiment20/normal_holdout');root=Path('artifacts/experiment20/features/R04');cfg=json.loads(Path('configs/experiment20.json').read_text());split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];hold=json.loads((out/'normal_holdout.json').read_text());q={(r['variant'],r['held_out_sequence']):r['normal_q99'] for r in hold['folds']};rows=[];removed=0
    with threadpool_limits(limits=4):
        model=LognormalDwellBaseline(cfg,load_process(cfg));model.fit([load_cache(root,s) for s in split['fit']])
        for seq in split['calibration']:
            d=load_cache(root,seq);a=load(art/f'19_exclude_{seq}_scores.npz');b=load(art/f'20_exclude_{seq}_scores.npz');refs=load(art/f'20_exclude_{seq}_references.npz');old=a['combined']>q['19',seq];new=b['combined']>q['20',seq];removed+=int(np.sum(old&~new));raw=model.raw(d)[0]
            for index in np.flatnonzero(new&~old):
                reasons=[]
                for role,frames,residuals in raw:
                    for j in np.flatnonzero(frames==index):
                        route=int(model.appearance_routes(d,role,frames[j:j+1])[0]);r=refs[f'route_calibration_{role}_{route}'];all_role=refs[f'calibration_{role}'];value=float(residuals[j]);s=pct(r,value)
                        if s>q['20',seq]:reasons.append({'role':role,'route':'phase' if route else 'pooled','raw_residual':value,'route_reference_max':float(r.max()),'role_reference_max':float(all_role.max()),'route_reference_samples':len(r),'route_percentile':s,'role_percentile':pct(all_role,value),'beyond_route_max':bool(value>r.max()),'within_role_range':bool(value<=all_role.max())})
                rows.append({'sequence':seq,'source_frame_index':int(d['indices'][index]),'relation_observed':bool(d['relation_valid'][index]),'combined_before':float(a['combined'][index]),'combined_after':float(b['combined'][index]),'new_visual_exceedances':reasons})
    result={'normal_holdout_only':True,'added_alarm_samples':len(rows),'removed_alarm_samples':removed,'added_alarms':rows,'all_extra_samples_have_route_tail_crossing_inside_role_range':all(any(r['beyond_route_max'] and r['within_role_range'] for r in row['new_visual_exceedances']) for row in rows),'note':'Descriptive check of fixed holdout predictions. Different normal q99 values retained. No model adjustment.'}
    (out/'normal_tail_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
