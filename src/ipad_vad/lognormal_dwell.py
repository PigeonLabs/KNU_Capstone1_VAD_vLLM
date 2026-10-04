"""Continuous lognormal duration scores fitted only to complete normal runs."""
import numpy as np
from scipy.special import ndtr
from ipad_vad.completed_dwell import CompletedContextDwell, CompletedDwellBaseline
from ipad_vad.context_dwell import observed_entry_context


class LognormalContextDwell(CompletedContextDwell):
    def __init__(self,sigma_floor=.05,**kwargs):
        super().__init__(**kwargs)
        if not np.isfinite(sigma_floor) or sigma_floor<=0:raise ValueError('Positive finite log sigma floor required')
        self.sigma_floor=sigma_floor

    def fit(self,caches):
        super().fit(caches);self.log_parameters={}
        for key,durations in self.context_durations.items():
            if not np.all(np.isfinite(durations)&(durations>0)):raise ValueError('Positive finite completed lengths required')
            logs=np.log(durations)
            self.log_parameters[key]=(float(logs.mean()),float(max(logs.std(ddof=0),self.sigma_floor)))

    def score(self,data):
        _,valid,age,reason=self.raw(data);context=observed_entry_context(data);score=np.zeros(len(age))
        for (previous,state),(mu,sigma) in self.log_parameters.items():
            use=valid&(context==previous)&(data['phases']==state)&(age>0)
            score[use]=ndtr((np.log(age[use])-mu)/sigma)
        return score,valid,age,reason


class LognormalDwellBaseline(CompletedDwellBaseline):
    def __init__(self,config,process):
        super().__init__(config,process)
        self.dwell=LognormalContextDwell(**config['normal_dwell'],**config['dwell_context'],**config['dwell_lognormal'])
