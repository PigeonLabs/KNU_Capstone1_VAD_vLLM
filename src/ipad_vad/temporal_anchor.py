"""Causal same-track aggregation of role margins, without filling missing boxes."""
from collections import deque
import numpy as np
from ipad_vad.verified_anchor import VerifiedAnchorPhase


class TemporalAnchorPhase(VerifiedAnchorPhase):
    def __init__(self,text_features,window=3,**kwargs):
        super().__init__(text_features,**kwargs)
        if not isinstance(window,int) or isinstance(window,bool) or window<1:raise ValueError('Positive integer window required')
        self.window=window

    def raw_margins(self,data):
        return super().margins(data)

    def margins(self,data):
        raw=self.raw_margins(data);result=raw.copy();history={};previous={};seen=set()
        ids=np.flatnonzero(data['roles']==self.anchor_role)
        for i in ids[np.argsort(data['object_frames'][ids],kind='stable')]:
            step=int(data['object_frames'][i]);track=int(data['tracks'][i]);key=(step,track)
            if key in seen:raise ValueError('Duplicate anchor track in one sample')
            seen.add(key)
            if track not in previous or step!=previous[track]+1:history[track]=deque(maxlen=self.window)
            history[track].append(raw[i]);previous[track]=step;result[i]=np.median(history[track])
        return result
