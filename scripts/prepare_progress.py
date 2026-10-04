"""Derive causal progress from frozen experiment03 anchors without labels."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from ipad_vad.spatial_phase import SpatialPhase
from ipad_vad.kinematics import progress_signal


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    p=argparse.ArgumentParser();p.add_argument('--normal-diagnostic-only',action='store_true')
    p.add_argument('--config',type=Path,default=Path('configs/experiment04.json'));args=p.parse_args()
    cfg=json.loads(args.config.read_text());scene=cfg['scene'];experiment=cfg['experiment'];lag=cfg.get('progress_lag_samples',1)
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    evidence=json.loads(Path('results/experiment03/grounding.json').read_text())['frozen_phase_model']
    phase=SpatialPhase(**cfg['spatial_phase']);phase.axis=0 if evidence['dominant_axis']=='x' else 1
    phase.band=np.array(evidence['perpendicular_band']);phase.centers=np.array(evidence['phase_centers'])
    source=Path(f'artifacts/experiment03/features/{scene}');target=Path(f'artifacts/experiment{experiment}/features/{scene}')
    out=Path(f'results/experiment{experiment}');out.mkdir(parents=True,exist_ok=True)
    paths=[source/f'training_{s}.npz' for s in split['fit']] if args.normal_diagnostic_only else sorted(source.glob('*.npz'))
    rows=[];phase_values={};hashes={}
    for path in paths:
        d=load(path);states,observed,chosen,_=phase.transform(d)
        assert np.array_equal(states,d['phases']),path
        velocity,valid,reason=progress_signal(d,chosen,phase.axis,lag_samples=lag)
        part,seq=path.stem.split('_');group='test' if part=='testing' else 'fit' if seq in split['fit'] else 'calibration'
        rows.append({'sequence_key':path.stem,'group':group,'samples':len(valid),'valid_progress_samples':int(valid.sum()),
                     'reason_counts':np.bincount(reason,minlength=5).tolist()})
        if group=='fit':
            for state in np.unique(states[valid]):phase_values.setdefault(int(state),[]).extend(velocity[valid&(states==state)].tolist())
        if not args.normal_diagnostic_only:
            target.mkdir(parents=True,exist_ok=True);original=set(d)
            d.update(motion_velocity=velocity,motion_valid=valid,motion_reason=reason,motion_detection_index=chosen)
            np.savez_compressed(target/path.name,**d)
            check=load(target/path.name)
            assert all(np.array_equal(d[k],check[k]) for k in original)
            hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    groups={}
    for group in ('fit','calibration','test'):
        subset=[r for r in rows if r['group']==group]
        if subset:
            n=sum(r['samples'] for r in subset);v=sum(r['valid_progress_samples'] for r in subset)
            groups[group]={'samples':n,'valid_progress_samples':v,'valid_fraction':v/n,'reason_counts':np.sum([r['reason_counts'] for r in subset],axis=0).tolist()}
    summary={'progress_lag_samples':lag,'mode':'normal_fit_only' if args.normal_diagnostic_only else 'all_label_blind','time_unit':'normalized_coordinate_per_source_frame',
             'groups':groups,'fit_phase_velocity':{str(k):{'samples':len(v),'quantiles_01_25_50_75_99':np.quantile(v,[.01,.25,.5,.75,.99]).tolist()} for k,v in sorted(phase_values.items())},
             'reason_codes':['valid','insufficient_history','no_current_anchor','history_anchor_absent','track_ID_change'],
             'invalid_policy':'velocity storage sentinel zero is masked; never fitted/calibrated/scored as a true zero speed',
             'sequences':rows,'source_sha256':hashes,'all_original_arrays_preserved':not args.normal_diagnostic_only}
    name='normal_progress_diagnostic.json' if args.normal_diagnostic_only else 'progress_features.json'
    (out/name).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('mode','groups','fit_phase_velocity')},indent=2))

if __name__=='__main__':main()
