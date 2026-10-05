"""Small shared metric adapter and conservative causal association fallback."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .tracking import iou_matrix

class ResidualMetric(nn.Module):
    def __init__(self, dim=512, rank=8):
        super().__init__();self.down=nn.Linear(dim,rank,bias=False);self.up=nn.Linear(rank,dim,bias=False);nn.init.zeros_(self.up.weight)
    def forward(self,x):
        x=F.normalize(x,dim=-1);return F.normalize(x+self.up(self.down(x)),dim=-1)

def pair_terms(model,a,b,negative_margin=.5):
    a=F.normalize(a,dim=-1);b=F.normalize(b,dim=-1);za,zb=model(a),model(b);cos=(za*zb).sum(-1)
    return 1-cos,F.relu(cos-negative_margin).square(),((za-a).square().sum(-1)+(zb-b).square().sum(-1))/2

class AssociationTracker:
    """IoU assignments are unchanged; only unmatched nodes use a supported metric.

    Per-role gates must be calibrated outside this class. Missing/disabled gates
    reduce EXACTLY to the historical IoU tracker. No prediction boxes are emitted.
    """
    def __init__(self,gates,threshold=.2,max_age=2,confidence=.5,max_center_distance=.25,max_area_ratio=4.,ambiguity_margin=.05):
        self.gates=gates;self.threshold=threshold;self.max_age=max_age;self.confidence=confidence;self.max_center_distance=max_center_distance;self.max_area_ratio=max_area_ratio;self.ambiguity_margin=ambiguity_margin;self.active={};self.next_id=0;self.edges=[]
    def update(self,boxes,roles,sample_index,features,confidence,detection_indices=None):
        boxes=np.asarray(boxes);roles=np.asarray(roles);features=np.asarray(features);confidence=np.asarray(confidence);self.active={k:v for k,v in self.active.items() if sample_index-v['last']<=self.max_age};ids=np.full(len(boxes),-1,dtype=np.int64)
        for role in set(roles):
            current=np.flatnonzero(roles==role);prior=[k for k,v in self.active.items() if v['role']==role]
            if not prior:continue
            overlap=iou_matrix([self.active[k]['box'] for k in prior],boxes[current]);cost=np.where(overlap>=self.threshold,1-overlap,1e6)
            for i,j in zip(*linear_sum_assignment(cost)):
                if overlap[i,j]>=self.threshold:ids[current[j]]=prior[i]
            gate=self.gates.get(int(role))
            if gate is None or gate>1:continue
            current=np.array([j for j in current if ids[j]<0 and confidence[j]>=self.confidence],dtype=int);prior=[k for k in prior if k not in ids and self.active[k]['confidence']>=self.confidence]
            if not len(current) or not prior:continue
            pb=np.array([self.active[k]['box'] for k in prior]);cb=boxes[current];dist=np.linalg.norm(((pb[:,:2]+pb[:,2:])/2)[:,None,:]-(cb[:,:2]+cb[:,2:])[None,:,:]/2,axis=-1);pa=np.maximum(pb[:,2:]-pb[:,:2],1e-9).prod(-1);ca=np.maximum(cb[:,2:]-cb[:,:2],1e-9).prod(-1);ratio=pa[:,None]/ca[None,:]
            cosine=np.clip(np.array([self.active[k]['feature'] for k in prior])@features[current].T,-1,1);valid=(dist<=self.max_center_distance)&(ratio<=self.max_area_ratio)&(ratio>=1/self.max_area_ratio)&(cosine>=gate)
            # Ambiguity is measured against other geometrically feasible candidates,
            # even if those fall just below the acceptance threshold.
            feasible=(dist<=self.max_center_distance)&(ratio<=self.max_area_ratio)&(ratio>=1/self.max_area_ratio)
            for i,j in zip(*np.where(valid.copy())):
                row=cosine[i,feasible[i]& (np.arange(len(current))!=j)];col=cosine[feasible[:,j]&(np.arange(len(prior))!=i),j]
                if (len(row) and cosine[i,j]-row.max()<self.ambiguity_margin) or (len(col) and cosine[i,j]-col.max()<self.ambiguity_margin):valid[i,j]=False
            cost=np.where(valid,1-cosine,1e6)
            for i,j in zip(*linear_sum_assignment(cost)):
                if not valid[i,j]:continue
                k=prior[i];index=current[j];ids[index]=k;old=self.active[k];self.edges.append({'role':int(role),'previous_sample':int(old['last']),'sample':int(sample_index),'previous_detection':int(old['detection']),'detection':int(detection_indices[index]) if detection_indices is not None else int(index),'track':int(k),'cosine':float(cosine[i,j]),'gate':float(gate),'center_distance':float(dist[i,j]),'area_ratio':float(ratio[i,j])})
        for j,(box,role) in enumerate(zip(boxes,roles)):
            if ids[j]<0:ids[j]=self.next_id;self.next_id+=1
            self.active[int(ids[j])]={'box':box.copy(),'role':role,'last':sample_index,'feature':features[j].copy(),'confidence':float(confidence[j]),'detection':int(detection_indices[j]) if detection_indices is not None else j}
        return ids
