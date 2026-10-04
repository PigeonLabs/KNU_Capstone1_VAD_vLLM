"""Normal-only relational latent states; semantics are not ground-truth action labels."""
from collections import deque
import numpy as np
from sklearn.cluster import KMeans


class RelationalPhase:
    names=['relative_center_x','relative_center_y','log_area_ratio','iou','anchor_center_x','anchor_center_y']

    def __init__(self, anchor_role=0, target_role=1, k=4, area_iqr_multiplier=1.5,
                 scale_floor=.05, smoothing=3, seed=42, reset_on_track_change=False):
        self.anchor_role=anchor_role;self.target_role=target_role;self.k=k
        self.area_iqr_multiplier=area_iqr_multiplier;self.scale_floor=scale_floor
        self.smoothing=smoothing;self.seed=seed
        if type(reset_on_track_change) is not bool:raise ValueError('Track reset must be boolean')
        self.reset_on_track_change=reset_on_track_change

    def fit_gates(self,caches):
        self.area_upper={};self.gate_evidence={}
        for role in (self.anchor_role,self.target_role):
            arrays=[]
            for d in caches:
                b=d['boxes'][d['roles']==role];area=np.maximum(b[:,2:]-b[:,:2],0).prod(1)
                arrays.append(area[area>0])
            areas=np.concatenate(arrays)
            if not len(areas):raise ValueError('Required role has no valid normal boxes')
            q1,q3=np.quantile(np.log(areas),[.25,.75])
            upper=float(min(1.,np.exp(q3+self.area_iqr_multiplier*(q3-q1))))
            self.area_upper[role]=upper
            self.gate_evidence[str(role)]={'normal_boxes':len(areas),'area_upper':upper,'excluded_fraction':float(np.mean(areas>upper+1e-8))}

    def select_candidate(self,data,ids,prior,role):
        """Choose among already eligible candidates; ties retain cache order."""
        same=ids[data['tracks'][ids]==prior];pool=same if len(same) else ids
        return int(pool[np.argmax(data['confidence'][pool])]) if len(pool) else -1

    def descriptors(self,data):
        n=len(data['indices']);features=np.zeros((n,6));valid=np.zeros(n,bool);chosen=np.full((n,2),-1,int)
        prior=[None,None];history=deque(maxlen=self.smoothing);previous_pair=None
        for step in range(n):
            for j,role in enumerate((self.anchor_role,self.target_role)):
                ids=np.flatnonzero((data['object_frames']==step)&(data['roles']==role))
                b=data['boxes'][ids];area=np.maximum(b[:,2:]-b[:,:2],0).prod(1)
                ids=ids[(area>0)&(area<=self.area_upper[role]+1e-8)]
                index=self.select_candidate(data,ids,prior[j],role)
                if index>=0:
                    chosen[step,j]=index
                    prior[j]=data['tracks'][chosen[step,j]]
            if np.any(chosen[step]<0):
                history.clear();previous_pair=None;continue
            pair=tuple(data['tracks'][chosen[step]])
            if self.reset_on_track_change and pair!=previous_pair:history.clear()
            previous_pair=pair
            a,b=data['boxes'][chosen[step]];size_a=a[2:]-a[:2];size_b=b[2:]-b[:2]
            center_a=(a[:2]+a[2:])/2;center_b=(b[:2]+b[2:])/2
            relative=(center_b-center_a)/np.maximum(size_a,1e-8)
            area_a=np.prod(size_a);area_b=np.prod(size_b)
            inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
            raw=np.r_[relative,np.log(area_b/area_a),inter/max(area_a+area_b-inter,1e-8),center_a]
            history.append(raw);features[step]=np.mean(history,axis=0);valid[step]=True
        return features,valid,chosen

    def fit(self,caches):
        self.fit_gates(caches);arrays=[];times=[]
        for d in caches:
            x,valid,_=self.descriptors(d);arrays.append(x[valid])
            times.append(d['indices'][valid]/max(int(d['frame_count'])-1,1))
        x=np.concatenate(arrays);time=np.concatenate(times)
        if len(x)<self.k*10:raise ValueError('Insufficient normal relation support')
        self.location=np.median(x,axis=0);q1,q3=np.quantile(x,[.25,.75],axis=0)
        self.scale=np.maximum(q3-q1,self.scale_floor)
        z=(x-self.location)/self.scale
        clustering=KMeans(n_clusters=self.k,random_state=self.seed,n_init=10).fit(z)
        if len(np.unique(clustering.labels_))!=self.k:raise ValueError('Collapsed relation clustering')
        median_time=[float(np.median(time[clustering.labels_==i])) for i in range(self.k)]
        order=np.argsort(median_time,kind='stable');self.centers=clustering.cluster_centers_[order]
        self.evidence={'descriptor_names':self.names,'area_gates':self.gate_evidence,'normal_valid_pairs':len(x),
            'median':self.location.tolist(),'scale_iqr_with_floor':self.scale.tolist(),'cluster_centers_scaled':self.centers.tolist(),
            'normal_cluster_samples':[int((clustering.labels_==i).sum()) for i in order],
            'median_normal_relative_position_for_label_permutation':[median_time[i] for i in order],
            'ordering_note':'Normal FIT relative frame position ONLY orders cluster IDs. It is not a phase label or a test-time input.',
            'semantic_note':'latent_0..k-1 have no verified mapping to actions or original VLM phases.'}

    def transform(self,data):
        x,valid,chosen=self.descriptors(data);phases=np.zeros(len(valid),int);last=0
        for i in range(len(valid)):
            if valid[i]:last=int(np.argmin(np.sum(((x[i]-self.location)/self.scale-self.centers)**2,axis=1)))
            phases[i]=last
        return phases,valid,chosen,x
