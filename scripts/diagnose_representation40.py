"""Per-process normal feature spread/drift, separate from detector accuracy."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from prepare_representation40 import OUT,ART,write
from experiment40_normal import RUNS


def main():
    manifest=json.loads((OUT/'data_manifest.json').read_text());rows=[];coverage=[]
    for scene,split in manifest['subsplits'].items():
        for part in ['representation_train','representation_validation','normal_calibration']:
            seqs=split[part];bank={};metadata=[]
            for run in RUNS:
                bank[run]={k:[] for k in ['global','crop']}
                for seq in seqs:
                    with np.load(ART/run/'features'/scene/f'training_{seq}.npz',allow_pickle=False) as f:
                        for kind in bank[run]:bank[run][kind].append(f[kind+'_features'])
                        if run=='B':
                            present=np.bincount(f['object_frames'],minlength=len(f['indices']))>0
                            metadata.append({'sequence':seq,'sampled_frames':len(present),'any_object_observed':int(present.sum()),'objects':len(f['boxes']),'phase_counts':np.bincount(f['phases']).tolist()})
                for kind in bank[run]:bank[run][kind]=np.concatenate(bank[run][kind])
            for run in RUNS:
                for kind,z in bank[run].items():
                    indices=np.linspace(0,len(z)-1,min(1024,len(z)),dtype=int);sample=z[indices].astype(np.float64);centered=sample-sample.mean(0);sv=np.linalg.svd(centered,compute_uv=False);energy=sv**2;prob=energy/energy.sum() if energy.sum()>0 else energy;rank=float(np.exp(-np.sum(prob[prob>0]*np.log(prob[prob>0])))) if energy.sum()>0 else 0
                    # CLIP and MobileCLIP coordinates differ: cosine drift for C/D vs B only.
                    drift=None if run=='A' else 1-np.sum(z*bank['B'][kind],axis=1)
                    rows.append({'scene':scene,'partition':part,'run':run,'kind':kind,'views':len(z),'spread_sample_views':len(sample),'sample_mean_feature_std':float(sample.std(0).mean()),'sample_effective_rank':rank,'cosine_distance_from_B_mean':None if drift is None else float(drift.mean()),'cosine_distance_from_B_p95':None if drift is None else float(np.quantile(drift,.95))})
            coverage.append({'scene':scene,'partition':part,'videos':len(metadata),'sampled_frames':sum(r['sampled_frames'] for r in metadata),'any_object_observed':sum(r['any_object_observed'] for r in metadata),'objects':sum(r['objects'] for r in metadata),'note':'Observed detections, not recall; no box/role/phase ground truth.'});print(scene,part,flush=True)
    write(OUT/'representation_diagnostic.json',{'normal_features':rows,'normal_detection_coverage':coverage,'note':'Canonical float32 normal features after all encoder training. Spread uses at most 1024 evenly indexed views per process/partition/kind. No anomaly labels or model selection. A/B cosine distance omitted because their embedding coordinates are not aligned. Effective rank is descriptive, not a detection metric.'})

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
