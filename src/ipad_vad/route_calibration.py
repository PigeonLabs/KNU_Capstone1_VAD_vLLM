"""Normal residual CDFs by object role and actual appearance-bank route."""
import numpy as np
from ipad_vad.scoring import empirical_percentile


def leave_one_video_out(sequence_ids):
    ids=list(sequence_ids)
    if len(ids)<2 or len(set(ids))!=len(ids):
        raise ValueError('At least two distinct normal sequence IDs required')
    return [(held,[s for s in ids if s!=held]) for held in ids]


class RouteCalibration:
    def __init__(self,minimum_observations=50,minimum_videos=2):
        for value in [minimum_observations,minimum_videos]:
            if not isinstance(value,int) or isinstance(value,bool) or value<1:
                raise ValueError('Positive integer support requirements required')
        self.minimum_observations=minimum_observations
        self.minimum_videos=minimum_videos
        self.references={};self.support={}

    def fit(self,rows):
        buckets={};videos={};self.references={};self.support={}
        for role,routes,residual,sequence_id in rows:
            if not isinstance(sequence_id,str) or not sequence_id:
                raise ValueError('Explicit nonempty sequence ID required')
            if routes.shape!=residual.shape or not np.isin(routes,[0,1]).all():
                raise ValueError('Routes must match residuals: 0 pooled, 1 phase')
            for route in np.unique(routes):
                key=(int(role),int(route));mask=routes==route
                buckets.setdefault(key,[]).append(residual[mask])
                videos.setdefault(key,set()).add(sequence_id)
        for key,parts in buckets.items():
            values=np.concatenate(parts)
            supported=len(values)>=self.minimum_observations and len(videos[key])>=self.minimum_videos
            self.support[key]={'observations':len(values),'videos':sorted(videos[key]),'supported':supported,
                               'midrank_step':1/(2*len(values))}
            if supported:self.references[key]=values

    def score(self,role,routes,residual,pooled_role_reference):
        score=empirical_percentile(pooled_role_reference,residual)
        for route in np.unique(routes):
            key=(int(role),int(route));mask=routes==route
            if key in self.references:
                score[mask]=empirical_percentile(self.references[key],residual[mask])
        return score
