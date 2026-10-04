"""Normal-trajectory grounding of R01 spatial phases, with causal observations."""
from collections import deque
import numpy as np
from sklearn.cluster import KMeans


class SpatialPhase:
    def __init__(self, role=0, k=3, min_track_samples=3, min_displacement=.1,
                 band_padding=.03, smoothing=3, seed=42):
        self.role=role;self.k=k;self.min_track_samples=min_track_samples
        self.min_displacement=min_displacement;self.band_padding=band_padding
        self.smoothing=smoothing;self.seed=seed

    def fit(self,caches):
        moving=[];track_count=0
        for data in caches:
            for track in np.unique(data['tracks'][data['roles']==self.role]):
                select=(data['tracks']==track)&(data['roles']==self.role)
                boxes=data['boxes'][select];centers=(boxes[:,:2]+boxes[:,2:])/2
                if len(centers)>=self.min_track_samples and np.ptp(centers,axis=0).max()>=self.min_displacement:
                    moving.append(centers);track_count+=1
        if not moving:raise ValueError('No sufficiently moving normal tracks; no silent fallback')
        points=np.concatenate(moving)
        self.axis=int(np.argmax(np.var(points,axis=0)));other=1-self.axis
        self.band=np.quantile(points[:,other],[.02,.98])+np.array([-self.band_padding,self.band_padding])
        self.centers=np.sort(KMeans(n_clusters=self.k,random_state=self.seed,n_init=10).fit(points[:,self.axis,None]).cluster_centers_.ravel())
        self.evidence={'fit_dynamic_tracks':track_count,'fit_dynamic_observations':len(points),
                       'dominant_axis':'x' if self.axis==0 else 'y','perpendicular_band':self.band.tolist(),
                       'phase_centers':self.centers.tolist(),
                       'note':'Phase indices follow increasing spatial coordinate for the R01 left/center/right descriptions; not a general learned action grammar.'}

    def transform(self,data):
        phases=[];observed=[];chosen=[];positions=[];history=deque(maxlen=self.smoothing)
        prior_track=None;last_phase=0
        for step in range(len(data['indices'])):
            candidates=np.flatnonzero((data['object_frames']==step)&(data['roles']==self.role))
            boxes=data['boxes'][candidates];centers=(boxes[:,:2]+boxes[:,2:])/2
            valid=(centers[:,1-self.axis]>=self.band[0])&(centers[:,1-self.axis]<=self.band[1])
            candidates=candidates[valid]
            if len(candidates):
                same=candidates[data['tracks'][candidates]==prior_track]
                pool=same if len(same) else candidates
                selected=int(pool[np.argmax(data['confidence'][pool])])
                box=data['boxes'][selected];position=float((box[self.axis]+box[self.axis+2])/2)
                history.append(position);prior_track=int(data['tracks'][selected])
                last_phase=int(np.argmin(np.abs(self.centers-np.mean(history))))
                observed.append(True);chosen.append(selected);positions.append(position)
            else:
                # No future interpolation; retain last phase, including initial phase 0.
                observed.append(False);chosen.append(-1);positions.append(float('nan'))
            phases.append(last_phase)
        return np.array(phases),np.array(observed),np.array(chosen),np.array(positions)
