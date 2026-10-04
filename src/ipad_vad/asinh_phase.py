"""Normal phase partitions in compressed coordinates under a frozen observation policy."""
import numpy as np
from sklearn.cluster import KMeans
from ipad_vad.confirmed_anchor import ConfirmedAnchorPhase


class AsinhConfirmedAnchorPhase(ConfirmedAnchorPhase):
    def phase_coordinates(self, descriptors):
        location=np.asarray(self.location);scale=np.asarray(self.scale)
        if location.shape!=(6,) or scale.shape!=(6,) or not np.isfinite(location).all() or not np.isfinite(scale).all() or np.any(scale<=0):
            raise ValueError('Finite frozen location and positive scale required')
        return np.arcsinh((descriptors-location)/scale)

    def fit(self,caches):
        """Fit centers only; caller must restore normal location/scale and area gates."""
        caches=list(caches)
        if not caches:raise ValueError('Normal FIT caches required')
        arrays=[];times=[];counts=[]
        for d in caches:
            x,valid,_=self.descriptors(d);arrays.append(x[valid]);counts.append(int(valid.sum()))
            times.append(d['indices'][valid]/max(int(d['frame_count'])-1,1))
        x=np.concatenate(arrays);time=np.concatenate(times)
        if len(x)<self.k*10:raise ValueError('Insufficient normal relation support')
        u=self.phase_coordinates(x)
        if not np.isfinite(u).all() or not np.isfinite(time).all():raise ValueError('Finite normal phase coordinates required')
        clustering=KMeans(n_clusters=self.k,random_state=self.seed,n_init=10).fit(u)
        if len(np.unique(clustering.labels_))!=self.k:raise ValueError('Collapsed relation clustering')
        median_time=[float(np.median(time[clustering.labels_==i])) for i in range(self.k)]
        order=np.argsort(median_time,kind='stable');self.centers=clustering.cluster_centers_[order]
        self.evidence={'descriptor_names':self.names,'coordinate_transform':'asinh','frozen_area_upper':{str(k):v for k,v in self.area_upper.items()},
            'normal_valid_pairs':len(x),'per_input_valid_pairs':counts,'median':self.location.tolist(),'scale_iqr_with_floor':self.scale.tolist(),
            'cluster_centers_transformed':self.centers.tolist(),'normal_cluster_samples':[int((clustering.labels_==i).sum()) for i in order],
            'median_normal_relative_position_for_label_permutation':[median_time[i] for i in order],
            'ordering_note':'Normal FIT relative position only orders latent IDs; it is not an inference input.',
            'semantic_note':'Gates, selection, raw descriptors and normal scaler are fixed. Clusters are not action labels.'}
        return self

    def transform(self,data):
        x,valid,chosen=self.descriptors(data);phases=np.zeros(len(valid),int);last=0
        u=self.phase_coordinates(x)
        for i in range(len(valid)):
            if valid[i]:last=int(np.argmin(np.sum((u[i]-self.centers)**2,axis=1)))
            phases[i]=last
        return phases,valid,chosen,x

    def save_phase(self,path):
        np.savez_compressed(path,coordinate_transform=np.array('asinh'),location=self.location,scale=self.scale,centers=self.centers,
                            area_roles=list(self.area_upper),area_upper=list(self.area_upper.values()))

    def load_phase(self,path):
        with np.load(path,allow_pickle=False) as f:d=dict(f)
        if str(d['coordinate_transform'])!='asinh':raise ValueError('Expected asinh phase model')
        if d['location'].shape!=(6,) or d['scale'].shape!=(6,) or d['centers'].shape!=(self.k,6):raise ValueError('Invalid phase parameter shape')
        if not all(np.isfinite(d[k]).all() for k in ['location','scale','centers','area_upper']) or np.any(d['scale']<=0) or np.any(d['area_upper']<=0):raise ValueError('Invalid phase parameters')
        if d['area_roles'].shape!=d['area_upper'].shape or set(d['area_roles'].tolist())!={self.anchor_role,self.target_role}:raise ValueError('Invalid frozen area roles')
        self.location=d['location'];self.scale=d['scale'];self.centers=d['centers'];self.area_upper=dict(zip(d['area_roles'].tolist(),d['area_upper'].tolist()))
        return self
