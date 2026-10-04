"""Add valid normal-duration evidence while preserving the transition branch."""
import numpy as np
from ipad_vad.process_calibration import StateCalibratedBaseline
from ipad_vad.dwell import NormalDwell


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
        result['process']=np.where(valid,np.maximum(transition,score),transition)
        result['combined']=self.fuse(result['visual'],result['process'])
        result.update(transition=transition,dwell=score,dwell_valid=valid,dwell_age=age,dwell_reason=reason)
        return result
