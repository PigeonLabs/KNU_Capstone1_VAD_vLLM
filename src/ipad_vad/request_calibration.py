"""Full-normal references for phase-request and explicit pooled-request policies."""
import numpy as np
from ipad_vad.scoring import observations,empirical_percentile


def request_residuals(model,data):
    """Canonical batches keep numerical ties independent of the chosen gate."""
    for role,frames,x in observations(data):
        phase_residual=np.empty(len(x))
        for phase in np.unique(data['phases'][frames]):
            mask=data['phases'][frames]==phase
            phase_residual[mask]=model.spaces[model.appearance_space_key(role,phase)].residual(x[mask])
        pooled=model.spaces[model.appearance_space_key(role,-1)].residual(x)
        yield role,frames,phase_residual,pooled


def calibration_codes(model,data,role,frames):
    mode=model.cfg.get('appearance_calibration_dispatch','request')
    if mode=='request':return (model.appearance_phases(data)[frames]>=0).astype(np.int8)
    if mode=='actual_bank':return model.appearance_routes(data,role,frames)
    raise ValueError(f'Unknown appearance calibration dispatch: {mode}')


def selected_residuals(model,data):
    phases=model.appearance_phases(data);requested=phases>=0
    if np.any(phases[requested]!=data['phases'][requested]):
        raise ValueError('Phase-request calibration requires the preserved source phase on phase requests')
    return [(role,frames,np.where(calibration_codes(model,data,role,frames)==1,phase,pool)) for role,frames,phase,pool in request_residuals(model,data)]


class RequestCalibration:
    def fit(self,model,caches):
        ids=[d['sequence_id'] for d in caches]
        if any(not isinstance(s,str) or not s for s in ids) or len(set(ids))!=len(ids):
            raise ValueError('Request calibration requires distinct explicit normal sequence IDs')
        refs={};videos={}
        for data,seq in zip(caches,ids):
            for role,_,phase,pool in request_residuals(model,data):
                for request,values in [(0,pool),(1,phase)]:
                    refs.setdefault((role,request),[]).append(values);videos.setdefault((role,request),set()).add(seq)
        self.references={key:np.concatenate(parts) for key,parts in refs.items()}
        if not self.references or any(not len(v) or not np.isfinite(v).all() for v in self.references.values()):
            raise ValueError('Request calibration needs finite nonempty normal references')
        self.support={key:{'observations':len(v),'videos':sorted(videos[key])} for key,v in self.references.items()}

    def score(self,role,requests,residual):
        requests=np.asarray(requests);residual=np.asarray(residual)
        if requests.shape!=residual.shape or not np.all(np.isin(requests,[0,1])):
            raise ValueError('Request codes must be matching 0/1 arrays')
        result=np.empty(len(residual))
        for request in np.unique(requests):
            key=(role,int(request))
            if key not in self.references:raise ValueError(f'No normal reference for request {key}')
            mask=requests==request;result[mask]=empirical_percentile(self.references[key],residual[mask])
        return result
