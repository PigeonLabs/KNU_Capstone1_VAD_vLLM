"""Normal complete-run duration evidence with causal, observed-entry inference."""
import numpy as np
from ipad_vad.scoring import empirical_percentile


def observed_ages(data):
    phases=data['phases'];indices=data['indices'];observed=data['relation_valid']
    ages=np.zeros(len(phases),dtype=float);known=np.zeros(len(phases),bool);entry=None
    for i in range(1,len(phases)):
        if not observed[i] or not observed[i-1]:entry=None
        elif phases[i]!=phases[i-1]:entry=indices[i]
        if entry is not None:
            ages[i]=indices[i]-entry;known[i]=True
    return ages,known


def complete_runs(data):
    phases=data['phases'];indices=data['indices'];observed=data['relation_valid']
    starts=np.r_[0,np.flatnonzero(phases[1:]!=phases[:-1])+1];ends=np.r_[starts[1:],len(phases)]
    for start,end in zip(starts,ends):
        # Both transitions and all observations inside the run must be observed.
        if start==0 or end==len(phases) or not observed[start-1:end+1].all():continue
        yield int(phases[start]),float(indices[end]-indices[start])


class NormalDwell:
    def __init__(self,minimum_complete_runs=10,minimum_calibration_samples=10):
        self.minimum_complete_runs=minimum_complete_runs;self.minimum_calibration_samples=minimum_calibration_samples
        if min(minimum_complete_runs,minimum_calibration_samples)<1:raise ValueError('Positive support required')

    def fit(self,caches):
        buckets={}
        for d in caches:
            for state,duration in complete_runs(d):buckets.setdefault(state,[]).append(duration)
        self.support={state:len(values) for state,values in buckets.items()}
        self.durations={state:np.sort(values) for state,values in buckets.items() if len(values)>=self.minimum_complete_runs}
        if not self.durations:raise ValueError('No supported normal dwell state')

    def raw(self,data):
        age,known=observed_ages(data);raw=np.zeros(len(age));valid=np.zeros(len(age),bool)
        reason=np.where(data['relation_valid'],2,1).astype(np.int8)
        reason[known]=3
        for state,values in self.durations.items():
            use=known&(data['phases']==state);n=len(values)
            count_ge=n-np.searchsorted(values,age[use],side='left')
            raw[use]=-np.log((1+count_ge)/(1+n));valid[use]=True;reason[use]=0
        return raw,valid,age,reason

    def calibrate(self,caches):
        values=[]
        for d in caches:
            raw,valid,_,_=self.raw(d);values.append(raw[valid])
        self.reference=np.concatenate(values)
        if len(self.reference)<self.minimum_calibration_samples:raise ValueError('Insufficient normal dwell calibration')

    def score(self,data):
        raw,valid,age,reason=self.raw(data);score=np.zeros(len(raw))
        score[valid]=empirical_percentile(self.reference,raw[valid])
        return score,valid,age,reason
