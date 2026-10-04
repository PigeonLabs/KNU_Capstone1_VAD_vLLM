"""Frozen-feature PCA and normal-only calibration for baseline experiment 01."""
import numpy as np
from sklearn.decomposition import PCA


class Subspace:
    def __init__(self, x, variance=.95, max_rank=32):
        x=np.asarray(x,dtype=np.float64)
        if len(x)<3:raise ValueError('At least three normal features required')
        self.mean=x.mean(0)
        # Keep at least one residual direction, including small phase groups.
        pca=PCA(n_components=min(max_rank,len(x)-2,x.shape[1]-1),svd_solver='full').fit(x)
        ratio=np.nan_to_num(pca.explained_variance_ratio_)
        rank=min(np.searchsorted(np.cumsum(ratio),variance)+1,len(pca.components_))
        self.basis=pca.components_[:rank];self.rank=int(rank);self.n=len(x)

    def residual(self,x):
        centered=np.asarray(x,dtype=np.float64)-self.mean
        return np.maximum(np.sum(centered**2,axis=-1)-np.sum((centered@self.basis.T)**2,axis=-1),0)


def empirical_percentile(reference, values):
    """Tie-aware percentile: a constant normal score maps to 0.5, not 1."""
    reference=np.sort(np.asarray(reference,dtype=float))
    if len(reference)==0:raise ValueError('Calibration reference cannot be empty')
    return (np.searchsorted(reference,values,side='left')+np.searchsorted(reference,values,side='right'))/(2*len(reference))


def observations(cache):
    # Role -1 is the full-frame branch. Tracks are scored by reusable object role.
    yield -1,np.arange(len(cache['indices'])),cache['global_features']
    for role in np.unique(cache['roles']):
        valid=cache['roles']==role
        yield int(role),cache['object_frames'][valid],cache['crop_features'][valid]


class Baseline:
    def __init__(self,config,process):
        self.cfg=config;self.process=process;self.k=len(process['phases'])
        self.spaces={};self.calibration={};self.fallback_counts={}

    def fit(self,caches):
        buckets={};counts=np.ones((self.k,self.k))*self.cfg['transition_laplace_alpha']
        for data in caches:
            phases=data['phases']
            np.add.at(counts,(phases[:-1],phases[1:]),1)
            for role,frames,x in observations(data):
                buckets.setdefault((role,-1),[]).append(x)
                for phase in np.unique(phases[frames]):
                    buckets.setdefault((role,int(phase)),[]).append(x[phases[frames]==phase])
        for key,arrays in buckets.items():
            x=np.concatenate(arrays)
            if len(x)>=self.cfg['minimum_phase_samples']:
                self.spaces[key]=Subspace(x,self.cfg['pca_variance'],self.cfg['pca_max_rank'])
        if (-1,-1) not in self.spaces:raise ValueError('Insufficient normal fit data')
        self.transition=counts/counts.sum(1,keepdims=True)
        ids={p['id']:i for i,p in enumerate(self.process['phases'])}
        order=[ids[s] for s in self.process['normal_order']]
        self.allowed=np.eye(self.k,dtype=bool)
        for a,b in zip(order[:-1],order[1:]):self.allowed[a,b]=True
        if self.process['cyclic']:self.allowed[order[-1],order[0]]=True

    def raw(self,data):
        phases=data['phases'];outputs=[]
        for role,frames,x in observations(data):
            residual=np.empty(len(x))
            for phase in np.unique(phases[frames]):
                key=(role,int(phase));fallback=False
                if key not in self.spaces:key=(role,-1);fallback=True
                if key not in self.spaces:key=(-1,-1);fallback=True
                if fallback:self.fallback_counts[str((role,int(phase)))]=self.fallback_counts.get(str((role,int(phase))),0)+int((phases[frames]==phase).sum())
                mask=phases[frames]==phase;residual[mask]=self.spaces[key].residual(x[mask])
            outputs.append((role,frames,residual))
        process=np.zeros(len(phases))
        if len(phases)>1:
            a,b=phases[:-1],phases[1:]
            process[1:]=-np.log(self.transition[a,b])+(~self.allowed[a,b]).astype(float)
        return outputs,process

    def calibrate(self,caches):
        raw=[self.raw(c) for c in caches]
        roles={role for obs,_ in raw for role,_,_ in obs}
        self.calibration={role:np.concatenate([r for obs,_ in raw for current,_,r in obs if current==role]) for role in roles}
        self.process_reference=np.concatenate([p for _,p in raw])
        self.fit_process_calibration(caches,raw)
        fused=np.concatenate([self.score(c)['combined'] for c in caches])
        self.threshold=float(np.quantile(fused,self.cfg['calibration_quantile'],method='higher'))

    def fit_process_calibration(self,caches,raw):
        """Extension point after global references exist, before fitting fused q99."""

    def calibrate_process(self,data,process):
        return empirical_percentile(self.process_reference,process)

    def fuse(self,visual,process):
        mode=self.cfg.get('score_fusion','mean')
        if mode=='max':return np.maximum(visual,process)
        if mode=='visual':return visual.copy()
        if mode!='mean':raise ValueError(f'Unknown score fusion: {mode}')
        w=self.cfg['visual_process_weight']
        return w*visual+(1-w)*process

    def score(self,data):
        raw,process=self.raw(data);visual=np.zeros(len(data['indices']));objects=[]
        for role,frames,residual in raw:
            # Missing calibration role has no justified percentile: explicitly fail.
            if role not in self.calibration:raise ValueError(f'Role {role} absent from normal calibration')
            score=empirical_percentile(self.calibration[role],residual)
            np.maximum.at(visual,frames,score)
            objects.append((role,frames,score))
        process=self.calibrate_process(data,process)
        return {'visual':visual,'process':process,'combined':self.fuse(visual,process),'objects':objects}
