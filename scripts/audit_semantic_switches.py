"""Posthoc normal metadata explanation of selection-switch reasons."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from experiment34_semantic_priority import OUT,root,load,SPLIT,write

def main():
    result={};records=[]
    for part in ['fit','calibration']:
        counts=Counter()
        for seq in SPLIT[part]:
            d=load(root('semantic')/f'training_{seq}.npz');old=load(root('control')/f'training_{seq}.npz');margin=d['anchor_gate_margin'];c=d['relation_detection_indices'];v=d['relation_valid'];upper=.21158000361228899
            for i in range(1,len(v)):
                if not(v[i-1] and v[i]):continue
                prev,now=c[i-1,0],c[i,0];pt=int(d['tracks'][prev]);nt=int(d['tracks'][now])
                if pt==nt:continue
                candidates=np.flatnonzero((d['object_frames']==i)&(d['roles']==1)&(d['tracks']==pt));reason='prior_cached_candidate_absent'
                if len(candidates):
                    assert len(candidates)==1;j=int(candidates[0]);area=float(np.maximum(d['boxes'][j,2:]-d['boxes'][j,:2],0).prod());area_ok=0<area<=upper+1e-8;sem_ok=margin[j]>0
                    reason='higher_margin_alternative' if area_ok and sem_ok else 'prior_area_excluded' if not area_ok and sem_ok else 'prior_semantic_excluded' if area_ok and not sem_ok else 'prior_both_excluded'
                    if reason=='higher_margin_alternative':assert margin[now]>margin[j]
                old_change=bool(old['relation_valid'][i-1] and old['relation_valid'][i] and old['tracks'][old['relation_detection_indices'][i-1,0]]!=old['tracks'][old['relation_detection_indices'][i,0]])
                bounce=bool(i+1<len(v) and v[i+1] and d['tracks'][c[i+1,0]]==pt)
                counts[reason]+=1;counts['new_anchor_change_boundaries']+=not old_change;counts['one_sample_return_to_prior_track']+=bounce
                records.append({'partition':part,'sequence':seq,'source_frame':int(d['indices'][i]),'previous_track':pt,'current_track':nt,'reason':reason,'also_changed_in_control':old_change,'next_valid_sample_returns_previous':bounce})
        result[part]=dict(counts)
    write(OUT/'normal_switch_mechanism.json',{'normal_only':True,'posthoc':True,'counts':result,'records':records,'one_sample_return_is_not_tracking_error_ground_truth':True});print(json.dumps(result,indent=2))

if __name__=='__main__':main()
