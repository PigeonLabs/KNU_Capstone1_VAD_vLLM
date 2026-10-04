"""Contrast intended and confounding roles before relation-only anchor selection."""
import numpy as np
from ipad_vad.relational_phase import RelationalPhase


class VerifiedAnchorPhase(RelationalPhase):
    def __init__(self,text_features,margin_threshold=0.,**kwargs):
        super().__init__(**kwargs)
        text=np.asarray(text_features,dtype=float)
        if text.ndim!=2 or text.shape[0]!=2 or not np.isfinite(text).all():raise ValueError('Two finite text embeddings required')
        norms=np.linalg.norm(text,axis=1,keepdims=True)
        if np.any(norms<=0):raise ValueError('Nonzero text embeddings required')
        self.text_features=text/norms;self.margin_threshold=float(margin_threshold)
        if not np.isfinite(self.margin_threshold):raise ValueError('Finite margin threshold required')

    def margins(self,data):
        features=np.asarray(data['crop_features'],dtype=float)
        if features.shape!=(len(data['roles']),self.text_features.shape[1]) or not np.isfinite(features).all():raise ValueError('Finite aligned crop features required')
        norms=np.linalg.norm(features,axis=1,keepdims=True)
        if np.any(norms<=0):raise ValueError('Nonzero crop features required')
        return (features/norms)@(self.text_features[0]-self.text_features[1])

    def phase_view(self,data):
        roles=data['roles'].copy()
        if np.any(roles<0):raise ValueError('Raw detected roles must be nonnegative')
        reject=(roles==self.anchor_role)&(self.margins(data)<=self.margin_threshold)
        roles[reject]=-999
        return dict(data,roles=roles)

    def fit_gates(self,caches):
        super().fit_gates([self.phase_view(d) for d in caches])

    def descriptors(self,data):
        return super().descriptors(self.phase_view(data))
