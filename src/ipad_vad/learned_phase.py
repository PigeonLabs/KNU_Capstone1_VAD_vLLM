"""Process-specific classifiers on shared frozen visual observations."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def causal_visual_input(data, window=3):
    """Global + highest-confidence crop per role, then a trailing visual mean.

    No geometry, phase IDs, sequence length, frame indices, or future features
    enter the numeric input. Empty role observations are explicit zero blocks.
    """
    global_x=np.asarray(data['global_features'],dtype=np.float32);n,dim=global_x.shape;rows=np.zeros((n,4*dim),dtype=np.float32);rows[:,:dim]=global_x
    for t in range(n):
        for role in range(3):
            ids=np.flatnonzero((data['object_frames']==t)&(data['roles']==role))
            if len(ids):i=ids[np.argmax(data['confidence'][ids])];rows[t,(role+1)*dim:(role+2)*dim]=data['crop_features'][i]
    rows/=np.maximum(np.linalg.norm(rows,axis=1,keepdims=True),1e-9);out=np.stack([rows[max(0,t-window+1):t+1].mean(0) for t in range(n)]);out/=np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-9);return out.astype(np.float32)


class PhaseHead(nn.Module):
    def __init__(self,classes,dim=2048,rank=8,adapt=False):
        super().__init__();self.classifier=nn.Linear(dim,classes);nn.init.normal_(self.classifier.weight,std=.01);nn.init.zeros_(self.classifier.bias);self.adapt=adapt
        if adapt:self.down=nn.Linear(dim,rank,bias=False);self.up=nn.Linear(rank,dim,bias=False);nn.init.zeros_(self.up.weight)
    def forward(self,x):
        z=F.normalize(x+self.up(self.down(x)),dim=-1) if self.adapt else x
        return self.classifier(z),z


def causal_hold(predicted,valid):
    predicted=np.asarray(predicted);valid=np.asarray(valid);assert predicted.ndim==valid.ndim==1 and len(predicted)==len(valid);out=np.zeros(len(valid),dtype=np.int64);last=0
    for i,(p,v) in enumerate(zip(predicted,valid)):
        if v:last=int(p)
        out[i]=last
    return out
