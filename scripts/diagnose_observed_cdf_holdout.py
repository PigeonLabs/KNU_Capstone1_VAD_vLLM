"""Post-evaluation attribution of paired normal holdout alarms; no parameter selection."""
import json
from pathlib import Path
import numpy as np
from audit_observed_transition_cdf import AFTER,PAIRS,load
from evaluate_route_holdout import load_cache
from ipad_vad.data import hold_scores


def main():
    out=Path('results/experiment28');art=Path('artifacts/experiment28');split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];rows=[]
    for e in AFTER:
        old=PAIRS[e]
        for held in split['calibration']:
            d=load_cache(art/'features/R04',held);a=load(art/'normal_holdout'/f'{old}_exclude_{held}_scores.npz');b=load(art/'normal_holdout'/f'{e}_exclude_{held}_scores.npz');ar=load(art/'normal_holdout'/f'{old}_exclude_{held}_references.npz');br=load(art/'normal_holdout'/f'{e}_exclude_{held}_references.npz');aa=a['combined']>ar['threshold'];bb=b['combined']>br['threshold'];dense_a=hold_scores(d['indices'],aa,int(d['frame_count']));dense_b=hold_scores(d['indices'],bb,int(d['frame_count']));changed=[]
            for i in np.flatnonzero(aa!=bb):
                previous=int(d['phases'][i-1]) if i>0 else -1;ref=br.get(f'process_reference_state_{previous}',np.empty(0));source='previous_state' if len(ref)>=10 else 'global_observed';selected=ref if len(ref)>=10 else br['process_reference'];raw=float(b['transition_raw'][i]);changed.append({'sample':int(i),'source_frame':int(d['indices'][i]),'end_frame_exclusive':int(d['indices'][i+1]) if i+1<len(aa) else int(d['frame_count']),'added':bool(bb[i]),'previous_state':previous,'current_state':int(d['phases'][i]),'observed_pair':bool(b['transition_valid'][i]),'state_reference_samples':len(ref),'selected_reference':source,'selected_reference_samples':len(selected),'selected_reference_max':float(np.max(selected)),'raw_transition':raw,'raw_above_reference_max':bool(raw>np.max(selected)),'old_transition_score':float(a['transition'][i]),'new_transition_score':float(b['transition'][i]),'visual_score':float(b['visual'][i])})
            rows.append({'variant':e,'held_out':held,'old_q99':float(ar['threshold']),'new_q99':float(br['threshold']),'added_dense_frames':int(np.sum(dense_b&~dense_a)),'removed_dense_frames':int(np.sum(dense_a&~dense_b)),'changed_samples':changed})
    (out/'normal_alarm_attribution.json').write_text(json.dumps({'normal_only':True,'post_evaluation_diagnostic':True,'rows':rows,'note':'Counterpart q99 and sample/dense scores are saved before test evaluation; no cutoff or reference is changed.'},indent=2)+'\n');print(json.dumps([r for r in rows if r['changed_samples']],indent=2))


if __name__=='__main__':main()
