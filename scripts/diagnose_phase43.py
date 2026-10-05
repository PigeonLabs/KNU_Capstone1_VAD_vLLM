"""Summarize normal phase support and temporal sensitivity, without action-GT claims."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from ipad_vad.learned_detector import write
from phase43_common import OUT,ART,SCENES,BRANCHES
from experiment41_phases import load_npz

def main():
    rows=[]
    for split in ['training','testing']:
        p=OUT/f'{split}_observations.json'
        if not p.exists():continue
        record=json.loads(p.read_text())
        for scene in SCENES:
            for branch in BRANCHES:
                for part in sorted(set(r['partition'] for r in record['sequences'] if r['scene']==scene)):
                    rr=[r for r in record['sequences'] if r['scene']==scene and r['branch']==branch and r['partition']==part];k=3 if scene=='R01' else 4;counts=np.array([r['phase_counts'] for r in rr]).sum(0);episodes=[e for r in rr for e in r['episodes']];complete=[e for e in episodes if e['status']=='complete'];rows.append({'split':split,'scene':scene,'branch':branch,'partition':part,'samples':sum(r['samples'] for r in rr),'valid_emissions':sum(r['emission_valid_samples'] for r in rr),'phase_counts':counts.tolist(),'phase_changed_vs_41':sum(r['phase_changed_vs_41'] for r in rr),'phase_changed_vs_teacher':sum(r['phase_changed_vs_teacher'] for r in rr),'transitions':sum(r['transitions'] for r in rr),'episode_status':dict(Counter(e['status'] for e in episodes)),'complete_episode_states':dict(Counter(str(e['phase']) for e in complete))})
    target=json.loads((OUT/'target_audit.json').read_text());support=[]
    for scene in SCENES:
        for part in ['representation_train','representation_validation']:
            rr=[r for r in target['sequences'] if r['scene']==scene and r['partition']==part];a=np.array([r['selected_state_counts'] for r in rr]);support.append({'scene':scene,'partition':part,'state_samples':a.sum(0).tolist(),'state_videos':(a>0).sum(0).tolist()})
    write(OUT/'phase_diagnostics.json',{'rows':rows,'target_support':support,'action_accuracy_measured':False,'note':'State counts and complete episodes are properties of each learned/weak partition, not verified physical action counts. Refit teacher must be compared separately from learned heads.'});print('Phase diagnostic rows',len(rows))
if __name__=='__main__':main()
