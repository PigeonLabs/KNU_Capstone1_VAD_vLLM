"""Read-only post-audit smoothing-window provenance; no alternative scores or cache updates."""
import json
from pathlib import Path
import numpy as np
from audit_transition_provenance import restore_model
from audit_missing_age import sha


def main():
    out=Path('results/experiment29');art=Path('artifacts/experiment29');cfg=json.loads(Path('configs/experiment29.json').read_text());model=restore_model(cfg);cases=json.loads((out/'case_selection.json').read_text())['cases'];case_ids={r['case_id'] for r in cases};totals={};details=[]
    for path in sorted((art/'traces').glob('*.json')):
        trace=json.loads(path.read_text());part=trace['partition'];frames=trace['frames'];summary=totals.setdefault(part,{'valid_samples':0,'mixed_pair_windows':0,'mixed_anchor_windows':0,'mixed_target_windows':0,'phases':{str(p):{'valid_samples':0,'mixed_pair_windows':0} for p in range(4)}})
        for i,frame in enumerate(frames):
            if not frame['relation_valid']:continue
            history=frames[i-frame['smoothing_observations']+1:i+1];pairs=[(r['selected']['anchor']['track'],r['selected']['target']['track']) for r in history];mixed=len(set(pairs))>1;summary['valid_samples']+=1;summary['mixed_pair_windows']+=mixed;summary['mixed_anchor_windows']+=len({p[0] for p in pairs})>1;summary['mixed_target_windows']+=len({p[1] for p in pairs})>1;summary['phases'][str(frame['phase'])]['valid_samples']+=1;summary['phases'][str(frame['phase'])]['mixed_pair_windows']+=mixed
        for row in trace['transitions']:
            if row['case_id'] not in case_ids:continue
            info=[]
            for i in [row['sample']-1,row['sample']]:
                f=frames[i];window=frames[i-f['smoothing_observations']+1:i+1];raw=np.array(f['raw_relation_descriptor']);nearest=int(np.argmin(np.sum(((raw-model.location)/model.scale-model.centers)**2,axis=1))) if f['relation_valid'] else None
                info.append({'source_frame':f['source_frame'],'stored_phase':f['phase'],'raw_descriptor_nearest_existing_center_diagnostic':nearest,'history_source_frames':[r['source_frame'] for r in window],'history_anchor_tracks':[r['selected']['anchor']['track'] for r in window],'history_target_tracks':[r['selected']['target']['track'] for r in window],'history_contains_multiple_pairs':len({(r['selected']['anchor']['track'],r['selected']['target']['track']) for r in window})>1,'stored_squared_distances':f['squared_distances_to_centers']})
            details.append({'case_id':row['case_id'],'frames':info})
    extra={}
    for frame in [144,368]:
        p=Path(cfg['data_root'])/f'R04/training/frames/02/{frame:03}.jpg';extra[str(p)]=sha(p)
    (out/'history_diagnostic.json').write_text(json.dumps({'normal_only':True,'post_selection_diagnostic':True,'totals':totals,'selected_cases':details,'additional_visual_followup_source_hashes':extra,'note':'Nearest-center values for unsmoothed raw descriptors are diagnostic only, not new phase assignments or score predictions. No test inputs or original caches/models modified.'},indent=2)+'\n');print(json.dumps(totals,indent=2));print(json.dumps([r for r in details if r['case_id'].startswith('calibration_02_0001') or r['case_id']=='calibration_02_000376'],indent=2))


if __name__=='__main__':main()
