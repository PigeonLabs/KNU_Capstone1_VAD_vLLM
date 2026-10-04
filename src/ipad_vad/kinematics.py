"""Causal, validity-aware progress features and normal phase-conditioned residuals."""
import numpy as np


def progress_signal(data, chosen, axis, lag_samples=1):
    """Signed normalized coordinate / source frame. Invalid values are masked, never stops."""
    if not isinstance(lag_samples,int) or isinstance(lag_samples,bool) or lag_samples<1:raise ValueError('lag_samples must be a positive integer')
    indices=np.asarray(data['indices']);chosen=np.asarray(chosen,dtype=int)
    if len(chosen)!=len(indices) or np.any(np.diff(indices)<=0):raise ValueError('Invalid sample chronology')
    velocity=np.zeros(len(indices),dtype=float);valid=np.zeros(len(indices),dtype=bool)
    # 0 valid; 1 insufficient history; 2 no current anchor; 3 missing history anchor; 4 ID change.
    reason=np.ones(len(indices),dtype=int)
    for step in range(lag_samples,len(indices)):
        window=chosen[step-lag_samples:step+1]
        a,b=window[0],window[-1]
        if b<0:reason[step]=2;continue
        if np.any(window<0):reason[step]=3;continue
        if np.any(data['tracks'][window]!=data['tracks'][b]):reason[step]=4;continue
        boxes=data['boxes'][[a,b]];centers=(boxes[:,axis]+boxes[:,axis+2])/2
        velocity[step]=(float(centers[1])-float(centers[0]))/int(indices[step]-indices[step-lag_samples])
        valid[step]=True;reason[step]=0
    return velocity,valid,reason


class NormalProgress:
    def __init__(self, minimum_samples=10, scale_floor=1e-6):
        self.minimum_samples=minimum_samples;self.scale_floor=scale_floor;self.models={}

    def fit(self,caches):
        values=np.concatenate([d['motion_velocity'][d['motion_valid']] for d in caches])
        phases=np.concatenate([d['phases'][d['motion_valid']] for d in caches])
        if len(values)<self.minimum_samples:raise ValueError('Insufficient valid normal motion')
        for phase in [-1,*np.unique(phases).tolist()]:
            x=values if phase==-1 else values[phases==phase]
            if len(x)<self.minimum_samples:continue
            median=float(np.median(x));scale=max(float(1.4826*np.median(np.abs(x-median))),self.scale_floor)
            self.models[int(phase)]={'median':median,'scale':scale,'samples':len(x)}

    def residual(self,data):
        valid=data['motion_valid'];raw=np.zeros(len(valid),dtype=float)
        for phase in np.unique(data['phases'][valid]):
            take=valid&(data['phases']==phase);m=self.models.get(int(phase),self.models[-1])
            raw[take]=np.abs(data['motion_velocity'][take]-m['median'])/m['scale']
        return raw,valid

    def calibrate(self,caches):
        residuals=[self.residual(d) for d in caches]
        self.reference=np.sort(np.concatenate([raw[valid] for raw,valid in residuals]))
        if not len(self.reference):raise ValueError('No valid calibration motion')

    def score(self,data):
        raw,valid=self.residual(data);score=np.zeros(len(raw),dtype=float)
        left=np.searchsorted(self.reference,raw[valid],side='left');right=np.searchsorted(self.reference,raw[valid],side='right')
        score[valid]=(left+right)/(2*len(self.reference))
        return score,valid
