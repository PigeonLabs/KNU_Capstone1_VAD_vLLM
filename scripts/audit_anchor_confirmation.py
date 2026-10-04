"""Explain normal switch mechanisms and confirmation delay without test labels."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from experiment35_confirmed_anchor import OUT,ART,root,load,SPLIT,write,models,independently_check


def main():
    relation=models();summary={};records=[];pending=[]
    for part in ['fit','calibration']:
        summary[part]={}
        for group in ['control','confirmed']:
            counts=Counter();decisions=Counter();next_reason=Counter();delays=[]
            for seq in SPLIT[part]:
                d=load(root(group)/f'training_{seq}.npz');m=relation[group];p,v,c,x=m.transform(d);trace=independently_check(d,m,group,p,v,c,x)
                assert trace==json.loads((ART/'selection_traces'/group/f'training_{seq}.json').read_text());decisions.update(r['reason'] for r in trace)
                for i,r in enumerate(trace):
                    if r['reason']=='pending_challenger':
                        outcome=trace[i+1]['reason'] if i+1<len(trace) else 'video_end';next_reason[outcome]+=1;pending.append(dict(r,partition=part,sequence=seq,next_reason=outcome))
                    if r['reason']=='confirmed_challenger':
                        assert i and trace[i-1]['reason']=='pending_challenger' and trace[i-1]['best_track']==r['best_track'] and r['confirmation_count']==2
                        delays.append(r['source_frame']-trace[i-1]['source_frame'])
                for i in range(1,len(v)):
                    if not(v[i-1] and v[i]):continue
                    oldid,newid=c[i-1,0],c[i,0];pt=int(d['tracks'][oldid]);nt=int(d['tracks'][newid])
                    if pt==nt:continue
                    ids=np.flatnonzero((d['object_frames']==i)&(d['roles']==1)&(d['tracks']==pt));reason='prior_cached_candidate_absent'
                    if len(ids):
                        assert len(ids)==1;j=int(ids[0]);area=float(np.maximum(d['boxes'][j,2:]-d['boxes'][j,:2],0).prod());area_ok=0<area<=m.area_upper[1]+1e-8;sem_ok=d['anchor_gate_margin'][j]>0
                        reason='eligible_challenger' if area_ok and sem_ok else 'prior_area_excluded' if not area_ok and sem_ok else 'prior_semantic_excluded' if area_ok and not sem_ok else 'prior_both_excluded'
                        if reason=='eligible_challenger':assert d['anchor_gate_margin'][newid]>d['anchor_gate_margin'][j]
                    bounce=bool(i+1<len(v) and v[i+1] and d['tracks'][c[i+1,0]]==pt);counts[reason]+=1;counts['one_sample_return_to_prior_track']+=bounce
                    records.append({'partition':part,'group':group,'sequence':seq,'source_frame':int(d['indices'][i]),'previous_track':pt,'current_track':nt,'reason':reason,'next_valid_sample_returns_previous':bounce})
            summary[part][group]={'consecutive_valid_anchor_switch_reasons':dict(counts),'all_sample_selection_decisions':dict(decisions),'pending_next_sample_reasons':dict(next_reason),'confirmed_delay_source_frames':delays,'confirmed_delay_meaning':'Delay from first pending winner observation; not ground-truth object/action recognition delay.'}
    write(OUT/'normal_confirmation_mechanism.json',{'normal_only':True,'posthoc':True,'summary':summary,'switches':records,'pending':pending,'tracking_or_role_accuracy_measured':False});print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
