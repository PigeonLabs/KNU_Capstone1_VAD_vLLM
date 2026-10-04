"""Post-evaluation normal-only population audit; no model or cutoff selection."""
import json
from pathlib import Path
import numpy as np
from audit_transition_gate import direct_mask
from evaluate_route_holdout import load_cache


def main():
    split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];root=Path('artifacts/experiment27/features/R04');rows={}
    for part in ['fit','calibration']:
        group={}
        for seq in split[part]:
            d=load_cache(root,seq);mask=direct_mask(d);group[seq]={'samples':len(mask),'consecutive_observed':int(mask.sum()),'by_previous_state':{str(s):{'all_transitions':int(np.sum(d['phases'][:-1]==s)),'consecutive_observed':int(np.sum((d['phases'][:-1]==s)&mask[1:]))} for s in range(4)}}
        rows[part]={'videos':group,'totals':{'samples':sum(v['samples'] for v in group.values()),'consecutive_observed':sum(v['consecutive_observed'] for v in group.values()),'by_previous_state':{str(s):{k:sum(v['by_previous_state'][str(s)][k] for v in group.values()) for k in ['all_transitions','consecutive_observed']} for s in range(4)}}}
    Path('results/experiment27/normal_transition_population.json').write_text(json.dumps({'normal_only':True,'post_evaluation_diagnostic':True,'partitions':rows,'note':'Existing transition references still include unobserved/held states. Counts describe evidence population, not semantic phase accuracy. No parameters fitted or selected.'},indent=2)+'\n');print(json.dumps({k:v['totals'] for k,v in rows.items()},indent=2))


if __name__=='__main__':main()
