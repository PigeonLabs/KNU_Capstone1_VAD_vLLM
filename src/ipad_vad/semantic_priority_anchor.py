"""Prefer temporal semantic evidence before continuity for eligible anchors."""
import numpy as np
from ipad_vad.temporal_anchor import TemporalAnchorPhase


class SemanticPriorityAnchorPhase(TemporalAnchorPhase):
    def phase_view(self,data):
        # Relation-only filtering; original roles/features remain unchanged.
        roles=data['roles'].copy()
        if np.any(roles<0):raise ValueError('Raw detected roles must be nonnegative')
        margins=self.margins(data)
        roles[(roles==self.anchor_role)&(margins<=self.margin_threshold)]=-999
        return dict(data,roles=roles,selection_margins=margins)

    def select_candidate(self,data,ids,prior,role):
        if role!=self.anchor_role or not len(ids):
            return super().select_candidate(data,ids,prior,role)
        margin=data['selection_margins'][ids]
        # Exact ties only: continuity, then confidence, then smallest cache index.
        top=ids[margin==np.max(margin)]
        return super().select_candidate(data,np.sort(top),prior,role)
