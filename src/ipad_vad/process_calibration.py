"""Normal calibration of transition scores conditional on the previous state."""
import numpy as np
from ipad_vad.scoring import Baseline, empirical_percentile
from ipad_vad.transition_evidence import observed_transition_mask


class StateCalibratedBaseline(Baseline):
    def fit_process_calibration(self,caches,raw):
        self.minimum_support=self.cfg['process_calibration']['minimum_support']
        if self.minimum_support<1:raise ValueError('Positive minimum support required')
        population=self.cfg['process_calibration'].get('population','all')
        if population not in ['all','consecutive_observed']:raise ValueError('Unknown transition calibration population')
        if population=='consecutive_observed':
            if self.cfg.get('transition_evidence_gate')!='consecutive_observed':
                raise ValueError('Observed transition calibration requires the matching evidence gate')
            masks=[observed_transition_mask(d) for d in caches]
            self.process_reference=np.concatenate([scores[mask] for (_,scores),mask in zip(raw,masks)])
            if len(self.process_reference)<self.minimum_support:raise ValueError('Insufficient global observed transition calibration support')
        else:masks=[np.r_[False,np.ones(max(len(d['phases'])-1,0),dtype=bool)][:len(d['phases'])] for d in caches]
        # Each state conditions on the previous sample; first samples never enter a state CDF.
        self.state_process_references={state:np.concatenate([
            scores[1:][(data['phases'][:-1]==state)&mask[1:]]
            for data,(_,scores),mask in zip(caches,raw,masks)]) for state in range(self.k)}

    def conditioning_mask(self,data):
        mask=np.zeros(len(data['phases']),dtype=bool)
        for state,ref in self.state_process_references.items():
            if len(ref)>=self.minimum_support:mask[1:]|=data['phases'][:-1]==state
        return mask

    def calibrate_process(self,data,process):
        score=super().calibrate_process(data,process)
        for state,ref in self.state_process_references.items():
            if len(ref)<self.minimum_support:continue
            positions=np.flatnonzero(data['phases'][:-1]==state)+1
            score[positions]=empirical_percentile(ref,process[positions])
        return score
