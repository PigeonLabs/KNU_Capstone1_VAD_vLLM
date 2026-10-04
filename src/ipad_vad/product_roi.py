"""Normal-motion ROI proposals; never use anomaly labels or future test frames."""
import numpy as np


def fit_product_roi(caches, role=0, min_track_samples=3, min_displacement=.1, padding=.03):
    moving=[]
    for data in caches:
        for track in np.unique(data['tracks'][data['roles']==role]):
            boxes=data['boxes'][(data['roles']==role)&(data['tracks']==track)]
            centers=(boxes[:,:2]+boxes[:,2:])/2
            if len(boxes)>=min_track_samples and np.ptp(centers,axis=0).max()>=min_displacement:
                moving.append(boxes)
    if not moving:raise ValueError('No normal moving tracks for ROI')
    boxes=np.concatenate(moving);centers=(boxes[:,:2]+boxes[:,2:])/2
    axis=int(np.argmax(np.var(centers,axis=0)));other=1-axis
    roi=np.array([0.,0.,1.,1.])
    roi[other]=max(0.,float(np.quantile(boxes[:,other],.02))-padding)
    roi[other+2]=min(1.,float(np.quantile(boxes[:,other+2],.98))+padding)
    return roi, {'dynamic_tracks':len(moving),'dynamic_boxes':len(boxes),'axis':axis,'roi':roi.tolist()}


def pixel_roi(roi, width, height):
    scale=np.array([width,height,width,height])
    scaled=np.asarray(roi)*scale
    return np.r_[np.floor(scaled[:2]),np.ceil(scaled[2:])].astype(int)


def restore_boxes(boxes, crop_rectangle, width, height):
    boxes=np.asarray(boxes).reshape(-1,4).copy()
    boxes+=np.tile(np.asarray(crop_rectangle)[:2],2)
    boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,width)
    boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,height)
    return boxes
