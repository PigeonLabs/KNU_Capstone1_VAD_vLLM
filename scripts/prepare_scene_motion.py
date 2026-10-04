"""Fit scene motion axis on normal FIT only, preserve semantic phase hypotheses."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from ipad_vad.experiment import load_process
from ipad_vad.motion_anchor import select_role_anchors,fit_motion_axis
from ipad_vad.kinematics import progress_signal


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--fit-only',action='store_true');args=p.parse_args()
    cfg=json.loads(args.config.read_text());scene=cfg['scene'];process=load_process(cfg);role=cfg['motion_role']
    if not 0<=role<len(process['objects']):raise ValueError('Invalid motion role')
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    source=Path('artifacts')/('experiment'+cfg['feature_source_experiment'])/'features'/scene
    target=Path('artifacts')/('experiment'+cfg['experiment'])/'features'/scene
    normal=[load(source/f'training_{s}.npz') for s in split['fit']]
    axis,evidence=fit_motion_axis(normal,role)
    paths=[source/f'training_{s}.npz' for s in split['fit']] if args.fit_only else sorted(source.glob('*.npz'))
    rows=[];fit_values=[];source_hashes={}
    for path in paths:
        d=load(path);chosen=select_role_anchors(d,role)
        velocity,valid,reason=progress_signal(d,chosen,axis,lag_samples=cfg['progress_lag_samples'])
        part,seq=path.stem.split('_');group='test' if part=='testing' else 'fit' if seq in split['fit'] else 'calibration'
        rows.append({'sequence_key':path.stem,'group':group,'samples':len(valid),'anchor_observed':int((chosen>=0).sum()),
                     'valid_motion':int(valid.sum()),'reason_counts':np.bincount(reason,minlength=5).tolist(),
                     'phase_counts':np.bincount(d['phases'],minlength=len(process['phases'])).tolist()})
        if group=='fit':fit_values.extend(velocity[valid].tolist())
        if not args.fit_only:
            target.mkdir(parents=True,exist_ok=True);original={k:v for k,v in d.items()}
            d.update(motion_velocity=velocity,motion_valid=valid,motion_reason=reason,motion_detection_index=chosen)
            np.savez_compressed(target/path.name,**d);check=load(target/path.name)
            assert all(np.array_equal(check[k],original[k]) for k in original),path
            source_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    values=np.array(fit_values);groups={}
    for group in ('fit','calibration','test'):
        subset=[r for r in rows if r['group']==group]
        if subset:
            total=sum(r['samples'] for r in subset)
            groups[group]={'samples':total,'observed_fraction':sum(r['anchor_observed'] for r in subset)/total,
                           'valid_motion_fraction':sum(r['valid_motion'] for r in subset)/total,
                           'phase_counts':np.sum([r['phase_counts'] for r in subset],axis=0).tolist()}
    if not len(values):raise ValueError('No valid normal motion')
    result={'scene':scene,'role':role,'role_id':process['objects'][role]['id'],'axis_model':evidence,'groups':groups,
            'fit_signed_velocity':{'samples':len(values),'quantiles_01_25_50_75_99':np.quantile(values,[.01,.25,.5,.75,.99]).tolist(),
                'positive_fraction':float(np.mean(values>1e-4)),'negative_fraction':float(np.mean(values< -1e-4)),
                'near_zero_fraction':float(np.mean(np.abs(values)<=1e-4)),'diagnostic_epsilon':1e-4},
            'sequences':rows,'source_sha256':source_hashes,'time_unit':'normalized_coordinate_per_source_frame',
            'fit_only':args.fit_only,'no_spatial_phase_or_roi_transfer':True,'all_source_arrays_preserved':not args.fit_only,
            'limitations':['Full object bbox center is not a tracked physical keypoint.','Signed pooled velocity can mix travel, rotation, standstill and box jitter.','No object/phase ground-truth accuracy claim.']}
    out=Path('results')/('experiment'+cfg['experiment']);out.mkdir(parents=True,exist_ok=True)
    (out/('normal_motion_diagnostic.json' if args.fit_only else 'motion_features.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('axis_model','groups','fit_signed_velocity')},indent=2))

if __name__=='__main__':main()
