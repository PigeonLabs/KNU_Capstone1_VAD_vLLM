"""Dwell percentile on complete FIT runs, without recalibrating observed ages."""
import numpy as np
from ipad_vad.context_dwell import ContextDwell, observed_entry_context
from ipad_vad.context_dwell_scoring import ContextDwellBaseline
from ipad_vad.scoring import empirical_percentile


class CompletedContextDwell(ContextDwell):
    def calibrate(self,caches):
        if not hasattr(self,'context_durations'):raise ValueError('Fit complete durations before scoring')
        # Kept as an explicitly empty serialization field: no age-CDF reference.
        self.reference=np.empty(0,dtype=float)

    def score(self,data):
        _,valid,age,reason=self.raw(data);context=observed_entry_context(data);score=np.zeros(len(age))
        for (previous,state),durations in self.context_durations.items():
            use=valid&(context==previous)&(data['phases']==state)
            score[use]=empirical_percentile(durations,age[use])
        return score,valid,age,reason


class CompletedDwellBaseline(ContextDwellBaseline):
    def __init__(self,config,process):
        super().__init__(config,process)
        if config['dwell_context']['distribution']!='entry':raise ValueError('Completed-run scoring requires entry durations')
        self.dwell=CompletedContextDwell(**config['normal_dwell'],**config['dwell_context'])
