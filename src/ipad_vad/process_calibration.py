"""Normal calibration of transition scores conditional on the previous state."""
import numpy as np
from ipad_vad.scoring import Baseline, empirical_percentile


class StateCalibratedBaseline(Baseline):
    def fit_process_calibration(self,caches,raw):
        self.minimum_support=self.cfg['process_calibration']['minimum_support']
        if self.minimum_support<1:raise ValueError('Positive minimum support required')
        # Exclude every sequence's first observation, which has no prior state.
        self.state_process_references={state:np.concatenate([
            scores[1:][data['phases'][:-1]==state]
            for data,(_,scores) in zip(caches,raw)]) for state in range(self.k)}

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
