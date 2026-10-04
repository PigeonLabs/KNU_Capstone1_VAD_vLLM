"""Simple class-gated Hungarian IoU tracker for the initial baseline."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    a,b=np.asarray(a).reshape(-1,4),np.asarray(b).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    intersection=np.maximum(hi-lo,0).prod(-1)
    area_a=np.maximum(a[:,2:]-a[:,:2],0).prod(-1)
    area_b=np.maximum(b[:,2:]-b[:,:2],0).prod(-1)
    return intersection/np.maximum(area_a[:,None]+area_b[None,:]-intersection,1e-9)


class Tracker:
    def __init__(self, threshold=.2, max_age=2):
        self.threshold=threshold;self.max_age=max_age;self.active={};self.next_id=0

    def update(self, boxes, roles, sample_index):
        self.active={k:v for k,v in self.active.items() if sample_index-v['last']<=self.max_age}
        ids=np.full(len(boxes),-1,dtype=np.int64)
        for role in set(roles):
            current=np.flatnonzero(np.asarray(roles)==role)
            prior=[k for k,v in self.active.items() if v['role']==role]
            if not prior:continue
            overlaps=iou_matrix([self.active[k]['box'] for k in prior],np.asarray(boxes)[current])
            # Gate before assignment, so invalid edges cannot displace valid matches.
            cost=np.where(overlaps>=self.threshold,1-overlaps,1e6)
            for i,j in zip(*linear_sum_assignment(cost)):
                if overlaps[i,j]>=self.threshold:ids[current[j]]=prior[i]
        for j,(box,role) in enumerate(zip(boxes,roles)):
            if ids[j]<0: ids[j]=self.next_id;self.next_id+=1
            self.active[int(ids[j])]={'box':box,'role':role,'last':sample_index}
        return ids
