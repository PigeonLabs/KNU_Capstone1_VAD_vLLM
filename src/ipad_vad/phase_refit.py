"""Refit latent relation partitions while preserving an existing observation policy."""
import numpy as np
from sklearn.cluster import KMeans


def refit_phase_centers(model, caches):
    """Use caller-supplied normal FIT videos only; never estimate detection gates."""
    caches=list(caches)
    if not caches:raise ValueError('Normal FIT caches required')
    if not hasattr(model,'area_upper'):raise ValueError('Frozen area gates required')
    for role in (model.anchor_role,model.target_role):
        if role not in model.area_upper or not np.isfinite(model.area_upper[role]) or model.area_upper[role]<=0:
            raise ValueError('Finite positive frozen gates required')
    arrays=[];times=[];counts=[]
    for d in caches:
        x,valid,_=model.descriptors(d);arrays.append(x[valid]);counts.append(int(valid.sum()))
        times.append(d['indices'][valid]/max(int(d['frame_count'])-1,1))
    x=np.concatenate(arrays);time=np.concatenate(times)
    if len(x)<model.k*10:raise ValueError('Insufficient normal relation support')
    if not np.isfinite(x).all() or not np.isfinite(time).all():raise ValueError('Finite normal descriptors required')
    location=np.median(x,axis=0);q1,q3=np.quantile(x,[.25,.75],axis=0);scale=np.maximum(q3-q1,model.scale_floor)
    z=(x-location)/scale
    clustering=KMeans(n_clusters=model.k,random_state=model.seed,n_init=10).fit(z)
    if len(np.unique(clustering.labels_))!=model.k:raise ValueError('Collapsed relation clustering')
    median_time=[float(np.median(time[clustering.labels_==i])) for i in range(model.k)]
    order=np.argsort(median_time,kind='stable');centers=clustering.cluster_centers_[order]
    # Commit learned values only after all support/finite/cluster checks succeed.
    model.location=location;model.scale=scale;model.centers=centers
    model.evidence={'descriptor_names':model.names,'frozen_area_upper':{str(k):v for k,v in model.area_upper.items()},
        'normal_valid_pairs':len(x),'per_input_valid_pairs':counts,'median':location.tolist(),
        'scale_iqr_with_floor':scale.tolist(),'cluster_centers_scaled':centers.tolist(),
        'normal_cluster_samples':[int((clustering.labels_==i).sum()) for i in order],
        'median_normal_relative_position_for_label_permutation':[median_time[i] for i in order],
        'ordering_note':'Normal FIT relative position only orders cluster IDs; inference never uses position.',
        'semantic_note':'Unsupervised clusters are not action ground truth. Gates and selection are unchanged.'}
    return model
