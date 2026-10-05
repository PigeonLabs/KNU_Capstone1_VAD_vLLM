"""Add valid normal-duration evidence while preserving the transition branch."""
import numpy as np
from ipad_vad.process_calibration import StateCalibratedBaseline
from ipad_vad.dwell import NormalDwell


from ipad_vad.transition_evidence import observed_transition_mask, same_track_pair_transition_mask


class DwellBaseline(StateCalibratedBaseline):
    def __init__(self,config,process):
        super().__init__(config,process);self.dwell=NormalDwell(**config['normal_dwell'])

    def fit(self,caches):
        super().fit(caches);self.dwell.fit(caches)

    def calibrate(self,caches):
        self.dwell.calibrate(caches)
        super().calibrate(caches)

    def score(self,data):
        result=super().score(data);score,valid,age,reason=self.dwell.score(data)
        transition=result['process'].copy()
        effective=transition
        gate=self.cfg.get('transition_evidence_gate')
        if gate is not None:
            if gate=='consecutive_observed':mask=observed_transition_mask(data)
            elif gate=='same_track_pair':mask=same_track_pair_transition_mask(data)
            else:raise ValueError('Unknown transition evidence gate')
            effective=np.where(mask,transition,0.)
            result.update(transition_raw=self.raw(data)[1],transition_valid=mask,transition_gated=effective)
        dwell_effective=np.where(valid,score,0.)
        dwell_gate=self.cfg.get('dwell_evidence_gate')
        if dwell_gate is not None:
            if dwell_gate not in ['ungated','same_track_pair_since_entry']:
                raise ValueError('Unknown dwell evidence gate')
            from ipad_vad.dwell_evidence import same_pair_since_entry_mask
            evidence=same_pair_since_entry_mask(data)
            if dwell_gate=='same_track_pair_since_entry':
                dwell_effective=np.where(evidence,dwell_effective,0.)
            result.update(dwell_evidence_valid=evidence,dwell_gated=dwell_effective)
        result['process']=np.maximum(effective,dwell_effective)
        result['combined']=self.fuse(result['visual'],result['process'])
        result.update(transition=transition,dwell=score,dwell_valid=valid,dwell_age=age,dwell_reason=reason)
        return result
