"""Post-result normal-only geometry and dwell evidence; no model selection."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from ipad_vad.context_dwell import complete_context_runs,observed_entry_context
from ipad_vad.dwell import observed_ages
from ipad_vad.transition_evidence import same_track_pair_transition_mask
from experiment37_asinh_phase import OUT,SPLIT,GROUPS,root,models,load,write,configure_common,normal_guard


def main():
    opened=normal_guard();audit=json.loads((OUT/'normal_audit.json').read_text());records=[];usage=[];summary={}
    for g in GROUPS:
        supported={tuple(k) for k in audit['variants'][f'37_{g}_hold']['supported_contexts']}
        for part in ['fit','calibration']:
            for seq in SPLIT[part]:
                d=load(root(g)/f'training_{seq}.npz');p=d['phases'];v=d['relation_valid'];same=same_track_pair_transition_mask(d);starts=np.r_[0,np.flatnonzero(p[1:]!=p[:-1])+1];ends=np.r_[starts[1:],len(p)];reconstructed=[]
                for start,end in zip(starts,ends):
                    if start==0 or end==len(p) or not v[start-1:end+1].all():continue
                    key=(int(p[start-1]),int(p[start]));duration=float(d['indices'][end]-d['indices'][start]);reconstructed.append((key,duration));incoming=bool(same[start]);within=bool(same[start+1:end].all());outgoing=bool(same[end]);pairs=d['tracks'][d['relation_detection_indices'][start-1:end+1]];assert bool(np.all(pairs==pairs[0]))==(incoming and within and outgoing)
                    records.append({'group':g,'partition':part,'sequence':seq,'context':f'{key[0]}->{key[1]}','supported':key in supported,'start':int(d['indices'][start]),'end':int(d['indices'][end]),'duration':duration,'incoming_same_pair':incoming,'within_same_pair':within,'outgoing_same_pair':outgoing,'identity_continuous_complete':incoming and within and outgoing})
                assert reconstructed==list(complete_context_runs(d))
                context=observed_entry_context(d);age,known=observed_ages(d);available=known&np.array([(int(a),int(b)) in supported for a,b in zip(context,p)]);identity=np.zeros(len(p),bool);ok=False
                for i in range(1,len(p)):
                    if not v[i] or not v[i-1]:ok=False
                    elif p[i]!=p[i-1]:ok=bool(same[i])
                    elif not same[i]:ok=False
                    identity[i]=ok
                usage.append({'group':g,'partition':part,'sequence':seq,'dwell_valid_samples':int(available.sum()),'same_pair_since_entry_samples':int(np.sum(available&identity))})
            rs=[r for r in records if r['group']==g and r['partition']==part];us=[r for r in usage if r['group']==g and r['partition']==part]
            summary[f'{g}_{part}']={'contexts':{c:{'runs':len(z),'supported':z[0]['supported'],'incoming_same_pair':sum(r['incoming_same_pair'] for r in z),'within_same_pair':sum(r['within_same_pair'] for r in z),'outgoing_same_pair':sum(r['outgoing_same_pair'] for r in z),'identity_continuous_complete':sum(r['identity_continuous_complete'] for r in z)} for c in sorted({r['context'] for r in rs}) for z in [[r for r in rs if r['context']==c]]},'usage':{k:sum(r[k] for r in us) for k in ['dwell_valid_samples','same_pair_since_entry_samples']}}
    m=models()['asinh'];fit=[load(root('asinh')/f'training_{s}.npz') for s in SPLIT['fit']];control=[load(root('control')/f'training_{s}.npz') for s in SPLIT['fit']];x=np.concatenate([d['relation_descriptors'][d['relation_valid']] for d in fit]);p=np.concatenate([d['phases'][d['relation_valid']] for d in fit]);oldp=np.concatenate([d['phases'][d['relation_valid']] for d in control]);z=(x-m.location)/m.scale;u=np.arcsinh(z);sparse=np.isin(oldp,[1,3]);geometry={}
    for name,coordinate in [('linear',z),('asinh',u)]:
        e=coordinate**2;geometry[name]={'squared_coordinate_energy_by_dimension':e.sum(0).tolist(),'dimension_fraction':(e.sum(0)/e.sum()).tolist(),'same_six_previous_rare_samples':int(sparse.sum()),'previous_rare_samples_fraction_of_squared_energy':float(e[sparse].sum()/e.sum()),'absolute_coordinate_max':np.abs(coordinate).max(0).tolist()}
    geometry['asinh']['within_cluster_squared_residual_by_dimension']=((u-m.centers[p])**2).sum(0).tolist()
    geometry['note']='Same fixed six normal FIT observations chosen by previous linear phases1/3. Different coordinate units; energy share is descriptive, not KMeans causal attribution. No alternative transform fitted.'
    write(OUT/'normal_duration_geometry_diagnostic.json',{'normal_only':True,'post_result_diagnostic':True,'model_or_score_changed':False,'summary':summary,'runs':records,'usage':usage,'geometry':geometry,'opened_data':sorted(opened)})
    print(json.dumps({'summary':summary,'geometry':geometry},indent=2),flush=True)

if __name__=='__main__':configure_common();main()
