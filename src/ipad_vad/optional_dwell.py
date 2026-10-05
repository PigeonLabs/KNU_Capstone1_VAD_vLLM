"""Explicit unavailable dwell evidence when normal support cannot identify it.

Legacy strict classes remain unchanged. Supported distributions/scores use the
legacy implementation. Unsupported evidence is masked, not declared normal.
"""
import numpy as np
from ipad_vad.dwell import complete_runs
from ipad_vad.context_dwell import complete_context_runs
from ipad_vad.lognormal_dwell import LognormalContextDwell,LognormalDwellBaseline


class OptionalLognormalContextDwell(LognormalContextDwell):
    def fit(self,caches):
        states={};contexts={}
        for data in caches:
            for key,duration in complete_runs(data):states.setdefault(key,[]).append(duration)
            for key,duration in complete_context_runs(data):contexts.setdefault(key,[]).append(duration)
        supported_states={k:np.sort(v) for k,v in states.items() if len(v)>=self.minimum_complete_runs}
        supported_contexts={k:np.sort(v) for k,v in contexts.items() if len(v)>=self.minimum_complete_runs}
        if supported_states and supported_contexts:
            super().fit(caches);self.available=True;self.unavailable_reason=None
        else:
            self.support={k:len(v) for k,v in states.items()};self.context_support={f'{a}->{b}':len(v) for (a,b),v in contexts.items()}
            self.durations=supported_states;self.context_durations={};self.log_parameters={}
            self.available=False;self.unavailable_reason='no_supported_normal_entry_context'


class OptionalLognormalDwellBaseline(LognormalDwellBaseline):
    def __init__(self,config,process):
        if config.get('dwell_allow_unavailable') is not True:raise ValueError('Explicit opt-in required for unavailable dwell policy')
        super().__init__(config,process)
        self.dwell=OptionalLognormalContextDwell(**config['normal_dwell'],**config['dwell_context'],**config['dwell_lognormal'])
