"""Normal FIT gap lengths and causal appearance fallback after observation loss."""
import numpy as np
from ipad_vad.fit_sampling import observed_mask


def checked_indices(data):
    indices=np.asarray(data['indices']);valid=observed_mask(data)
    if indices.shape!=valid.shape or indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer):
        raise ValueError('Source indices must be a matching one-dimensional integer array')
    if np.any(indices<0) or np.any(np.diff(indices)<=0):raise ValueError('Source indices must increase strictly')
    return indices,valid


def causal_age(data):
    indices,valid=checked_indices(data);age=np.full(len(indices),-1,dtype=np.int64);held=np.full(len(indices),-1,dtype=np.int64)
    last_index=None;last_phase=-1
    for i,(index,observed) in enumerate(zip(indices,valid)):
        if observed:last_index=int(index);last_phase=int(data['phases'][i])
        if last_index is not None:age[i]=int(index)-last_index;held[i]=last_phase
    return age,held


class MissingAge:
    def __init__(self,quantile=.9):
        if isinstance(quantile,bool) or not isinstance(quantile,(int,float)) or not 0<quantile<=1:
            raise ValueError('Gap quantile must lie in (0,1]')
        self.quantile=float(quantile);self.tau=None

    def fit(self,caches):
        self.tau=None;self.complete=[];self.censored=[]
        ids=[d['sequence_id'] for d in caches]
        if any(not isinstance(s,str) or not s for s in ids) or len(set(ids))!=len(ids):
            raise ValueError('Gap FIT requires distinct explicit sequence IDs')
        for d in caches:
            indices,valid=checked_indices(d);edges=np.diff(np.r_[False,~valid,False].astype(int))
            for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                row={'sequence_id':d['sequence_id'],'start_sample':int(start),'end_sample_exclusive':int(end),'samples':int(end-start),'start_frame':int(indices[start]),'last_missing_frame':int(indices[end-1])}
                if start>0 and end<len(indices):
                    assert valid[start-1] and valid[end]
                    row['duration_frames']=int(indices[end]-indices[start]);self.complete.append(row)
                else:
                    row.update(left_censored=bool(start==0),right_censored=bool(end==len(indices)),observed_span_lower_bound_frames=int(indices[end-1]-indices[start]));self.censored.append(row)
        self.durations=np.array([r['duration_frames'] for r in self.complete],dtype=np.int64)
        if not len(self.durations):raise ValueError('No complete normal FIT gaps; missing-age fallback unavailable')
        self.tau=int(np.quantile(self.durations,self.quantile,method='higher'))

    def states(self,data):
        if self.tau is None:raise RuntimeError('Fit missing-age threshold before inference')
        age,held=causal_age(data);valid=observed_mask(data)
        # 0 observed; 1 no prior observation; 2 missing within tau; 3 stale missing.
        state=np.where(valid,0,np.where(age<0,1,np.where(age<=self.tau,2,3))).astype(np.int8)
        return age,state,np.where((state==1)|(state==3),-1,held)

    def report(self):
        return {'quantile':self.quantile,'method':'higher','tau_frames':self.tau,'complete_runs':self.complete,'censored_runs':self.censored,'time_unit':'source_frame_index','fit_only':True}
