"""Post-result fallback context; no refitting or choice of scores/thresholds."""
import json
from pathlib import Path
import numpy as np
from ipad_vad.missing_age import causal_age
from ipad_vad.data import hold_scores
from experiment37_asinh_phase import OUT,SOURCE,SPLIT,GROUPS,GATES,root,load,write


def main():
    diagnostic=json.loads((OUT/'diagnostic.json').read_text());out={}
    for part,seqs in [('fit',SPLIT['fit']),('calibration',SPLIT['calibration']),('test',[p.stem.split('_')[1] for p in sorted(SOURCE.glob('testing_*.npz'))])]:
        rows=[];scores={}
        for seq in seqs:
            d=load(root('asinh')/f'{"testing" if part=="test" else "training"}_{seq}.npz');age,_=causal_age(d);v=d['relation_valid'];state=np.where(v,0,np.where(age<0,1,np.where(age<=56,2,3)));dense=hold_scores(d['indices'],state,int(d['frame_count']));rows.append({'sequence':seq,'sample_states':np.bincount(state,minlength=4).tolist(),'frame_states':np.bincount(dense,minlength=4).tolist()})
            if part=='test':
                for g in GROUPS:
                    for gate in GATES:
                        e=f'37_{g}_{gate}';q=diagnostic['variants'][e]['q99'];a=load(f'artifacts/experiment{e}/predictions/R04_{seq}.npz');scores.setdefault(e,{str(k):{'normal_frames':0,'anomaly_frames':0,'fp':0,'tp':0} for k in range(4)})
                        for k in range(4):
                            mask=dense==k;alarm=a['combined']>q;cell=scores[e][str(k)];cell['normal_frames']+=int(np.sum(mask&(a['labels']==0)));cell['anomaly_frames']+=int(np.sum(mask&(a['labels']==1)));cell['fp']+=int(np.sum(mask&alarm&(a['labels']==0)));cell['tp']+=int(np.sum(mask&alarm&(a['labels']==1)))
        out[part]={'sequences':rows,'sample_states':np.sum([r['sample_states'] for r in rows],axis=0).tolist(),'frame_states':np.sum([r['frame_states'] for r in rows],axis=0).tolist(),'test_alarms_by_state':scores or None}
    for e,ss in out['test']['test_alarms_by_state'].items():
        assert sum(v['fp'] for v in ss.values())==diagnostic['variants'][e]['alarms']['normal'];assert sum(v['tp'] for v in ss.values())==diagnostic['variants'][e]['alarms']['anomaly']
    write(OUT/'fallback_context.json',{'partitions':out,'state_names':{'0':'observed','1':'no_prior_observation','2':'missing_within_tau','3':'stale_missing'},'tau_source_frames':56,'note':'Posthoc descriptive context. Initial numeric phase0 is a supported appearance bank in both controls, but changes composition; hold can request it before first observation, whereas pool/age request pooled. Different full calibration CDFs also change scores elsewhere, so this is not causal effect isolation.'})
    print(json.dumps(out['test'],indent=2))

if __name__=='__main__':main()
