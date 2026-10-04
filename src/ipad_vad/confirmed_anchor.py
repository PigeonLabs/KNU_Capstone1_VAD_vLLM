"""Causal confirmation before replacing an anchor that remains eligible."""
import numpy as np
from ipad_vad.semantic_priority_anchor import SemanticPriorityAnchorPhase


class ConfirmedAnchorPhase(SemanticPriorityAnchorPhase):
    def __init__(self,*args,confirmation_samples=2,**kwargs):
        super().__init__(*args,**kwargs)
        if type(confirmation_samples) is not int or confirmation_samples<1:
            raise ValueError('Positive integer confirmation_samples required')
        self.confirmation_samples=confirmation_samples

    def phase_view(self,data):
        view=super().phase_view(data)
        # Local to this descriptor call/video, including calls made during fit.
        view['anchor_confirmation_state']={}
        return view

    def select_candidate(self,data,ids,prior,role):
        if role!=self.anchor_role:
            return super().select_candidate(data,ids,prior,role)
        state=data['anchor_confirmation_state']
        if not len(ids):
            state.clear();return -1
        best=super().select_candidate(data,ids,prior,role)
        same=ids[data['tracks'][ids]==prior]
        if not len(same) or data['tracks'][best]==prior:
            state.clear();return best
        step=int(data['object_frames'][best]);track=int(data['tracks'][best])
        count=state.get('count',0)+1 if state.get('track')==track and state.get('step')==step-1 else 1
        if count>=self.confirmation_samples:
            state.clear();return best
        state.update(track=track,step=step,count=count)
        return int(same[np.argmax(data['confidence'][same])])
