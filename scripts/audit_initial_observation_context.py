"""Post-result interval/identity provenance for initial routing; no score changes."""
import json
from pathlib import Path
import numpy as np
from ipad_vad.data import hold_scores
from ipad_vad.transition_evidence import same_track_pair_transition_mask
from experiment38_initial_observation import OUT,SOURCE,VARIANTS,load,write,scalar_seen


def main():
    diagnostic=json.loads((OUT/'diagnostic.json').read_text());rows=[];identity=[];prefixes=0;request_rows=[]
    for path in sorted(SOURCE.glob('*.npz')):
        d=load(path);seen=scalar_seen(d);part,seq=path.stem.split('_');n=len(seen)
        # Independent reference to the first valid index, used for audit only.
        valid=np.flatnonzero(d['relation_valid']);expected=np.arange(n)>=(valid[0] if len(valid) else n);np.testing.assert_array_equal(seen,expected)
        for end in [max(1,n//2),max(1,n-1)]:np.testing.assert_array_equal(scalar_seen({'relation_valid':d['relation_valid'][:end]}),seen[:end]);prefixes+=1
        if part!='testing':continue
        dense=hold_scores(d['indices'],~seen,int(d['frame_count'])).astype(bool);same=same_track_pair_transition_mask(d);p=d['phases'];v=d['relation_valid'];since=np.zeros(n,bool);known=False
        for i in range(1,n):
            if not v[i] or not v[i-1]:known=False
            elif p[i]!=p[i-1]:known=bool(same[i])
            elif not same[i]:known=False
            since[i]=known
        continuity=hold_scores(d['indices'],since,int(d['frame_count'])).astype(bool)
        for e in VARIANTS:
            pred=load(f'artifacts/experiment{e}/predictions/R04_{seq}.npz');q=diagnostic['variants'][e]['q99'];y=pred['labels'];alarm=pred['combined']>q;dwellonly=pred['dwell_valid']&(pred['dwell']>q)&~(pred['visual']>q)&~(pred['transition_gated']>q)
            rows.append({'variant':e,'sequence':seq,'initial_frames':int(dense.sum()),'fp_before':int(np.sum(dense&alarm&(y==0))),'tp_before':int(np.sum(dense&alarm&(y==1))),'fp_after':int(np.sum(~dense&alarm&(y==0))),'tp_after':int(np.sum(~dense&alarm&(y==1)))})
            if e=='38_guarded_hold':
                identity.append({'sequence':seq,'dwell_valid_frames':int(pred['dwell_valid'].sum()),'same_pair_since_entry_valid_frames':int(np.sum(pred['dwell_valid']&continuity)),'dwell_only_fp_with_continuity':int(np.sum(dwellonly&continuity&(y==0))),'dwell_only_tp_with_continuity':int(np.sum(dwellonly&continuity&(y==1))),'dwell_only_fp_without_continuity':int(np.sum(dwellonly&~continuity&(y==0))),'dwell_only_tp_without_continuity':int(np.sum(dwellonly&~continuity&(y==1)))})
    summary={e:{k:sum(r[k] for r in rows if r['variant']==e) for k in ['initial_frames','fp_before','tp_before','fp_after','tp_after']} for e in VARIANTS}
    for e,s in summary.items():assert s['fp_before']+s['fp_after']==diagnostic['variants'][e]['alarms']['normal'];assert s['tp_before']+s['tp_after']==diagnostic['variants'][e]['alarms']['anomaly']
    identity_summary={k:sum(r[k] for r in identity) for k in identity[0] if k!='sequence'}
    write(OUT/'initial_and_dwell_context.json',{'initial_mask_prefix_checks':prefixes,'source_features_checked':44,'test_region_summary':summary,'test_per_sequence':rows,'guarded_hold_dwell_identity_provenance':identity,'dwell_identity_summary':identity_summary,'note':'Post-result metadata audit only. No new gate or threshold applied. Same-track continuity is not identity/action ground truth; normal duration support is unchanged from experiment37.'})
    print(json.dumps({'regions':summary,'dwell_identity':identity_summary},indent=2),flush=True)

if __name__=='__main__':main()
