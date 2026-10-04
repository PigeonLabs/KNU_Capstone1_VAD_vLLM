"""Entry-conditioned dwell and a support-matched pooled-duration control."""
import numpy as np
from ipad_vad.dwell import NormalDwell, observed_ages


def observed_entry_context(data):
    phase=data['phases'];observed=data['relation_valid'];context=np.full(len(phase),-1,dtype=int);entry=-1
    for i in range(1,len(phase)):
        if not observed[i] or not observed[i-1]:entry=-1
        elif phase[i]!=phase[i-1]:entry=int(phase[i-1])
        context[i]=entry
    return context


def complete_context_runs(data):
    phase=data['phases'];indices=data['indices'];observed=data['relation_valid']
    starts=np.r_[0,np.flatnonzero(phase[1:]!=phase[:-1])+1];ends=np.r_[starts[1:],len(phase)]
    for start,end in zip(starts,ends):
        if start==0 or end==len(phase) or not observed[start-1:end+1].all():continue
        yield (int(phase[start-1]),int(phase[start])),float(indices[end]-indices[start])


class ContextDwell(NormalDwell):
    def __init__(self,distribution='entry',**kwargs):
        super().__init__(**kwargs)
        if distribution not in ('entry','pooled'):raise ValueError('Unknown dwell distribution')
        self.distribution=distribution

    def fit(self,caches):
        super().fit(caches);buckets={}
        for data in caches:
            for key,duration in complete_context_runs(data):buckets.setdefault(key,[]).append(duration)
        self.context_support={f'{a}->{b}':len(values) for (a,b),values in buckets.items()}
        self.context_durations={key:np.sort(values) for key,values in buckets.items() if len(values)>=self.minimum_complete_runs}
        if not self.context_durations:raise ValueError('No supported normal entry context')

    def raw(self,data):
        age,known=observed_ages(data);context=observed_entry_context(data)
        raw=np.zeros(len(age));valid=np.zeros(len(age),bool);reason=np.where(data['relation_valid'],2,1).astype(np.int8);reason[known]=3
        for (previous,state),values in self.context_durations.items():
            use=known&(context==previous)&(data['phases']==state)
            # The same context gate is used for both experimental arms.
            if self.distribution=='pooled':values=self.durations[state]
            count_ge=len(values)-np.searchsorted(values,age[use],side='left')
            raw[use]=-np.log((1+count_ge)/(1+len(values)));valid[use]=True;reason[use]=0
        return raw,valid,age,reason
