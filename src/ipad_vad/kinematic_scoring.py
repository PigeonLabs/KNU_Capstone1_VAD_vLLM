"""Experiment04 process branch adds calibrated motion evidence by a fixed max rule."""
import numpy as np
from ipad_vad.scoring import Baseline
from ipad_vad.kinematics import NormalProgress


class KinematicBaseline(Baseline):
    def __init__(self,config,process):
        super().__init__(config,process)
        self.motion=NormalProgress(**config['normal_progress'])

    def fit(self,caches):
        super().fit(caches);self.motion.fit(caches)

    def calibrate(self,caches):
        self.motion.calibrate(caches)
        # Base calibration calls this class's score, so q99 matches augmented fusion.
        super().calibrate(caches)

    def score(self,data):
        result=super().score(data);motion,valid=self.motion.score(data)
        transition=result['process'].copy()
        result['process']=np.where(valid,np.maximum(transition,motion),transition)
        w=self.cfg['visual_process_weight']
        result['combined']=w*result['visual']+(1-w)*result['process']
        result.update(motion=motion,motion_valid=valid,transition=transition)
        return result
