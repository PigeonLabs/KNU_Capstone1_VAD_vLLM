"""Verify a fixed fusion ablation and describe added/lost alarms against visual only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',default='09');parser.add_argument('--max-experiment',default='10');parser.add_argument('--visual-experiment',default='10_visual');args=parser.parse_args()
    ns=[args.baseline,args.max_experiment,args.visual_experiment];roots={n:Path(f'artifacts/experiment{n}') for n in ns}
    configs={n:json.loads(Path(f'configs/experiment{n}.json').read_text()) for n in ns};scene=configs[ns[0]]['scene']
    models={n:load(root/'normal_model.npz') for n,root in roots.items()};metrics={n:json.loads(Path(f'results/experiment{n}/metrics.json').read_text()) for n in ns}
    samples={};thresholds={n:metrics[n]['normal_q99_threshold'] for n in ns};all_predictions={n:[] for n in ns}
    baseline=roots[ns[0]];before=models[ns[0]]
    for n in ns[1:]:
        cfg=configs[n];after=models[n];assert set(before)==set(after)
        for key in before:
            if key!='threshold':np.testing.assert_array_equal(before[key],after[key],err_msg=key)
        protocol=json.loads(Path(f'results/experiment{n}/pre_evaluation_protocol.json').read_text())
        assert protocol['config_sha256']==hashlib.sha256(Path(f'configs/experiment{n}.json').read_bytes()).hexdigest()
        names=sorted(p.name for p in (baseline/'features'/scene).glob('*.npz'))
        assert names==sorted(p.name for p in (roots[n]/'features'/scene).glob('*.npz'))
        for name in names:
            a=(baseline/'features'/scene/name).read_bytes();b=(roots[n]/'features'/scene/name).read_bytes()
            assert a==b and hashlib.sha256(b).hexdigest()==protocol['source_sha256'][name]
        cal=load(roots[n]/'normal_calibration_scores.npz');samples[n]=cal
        expected=np.maximum(cal['visual'],cal['process']) if cfg['score_fusion']=='max' else cal['visual']
        np.testing.assert_array_equal(cal['combined'],expected)
        q=float(np.quantile(expected,cfg['calibration_quantile'],method='higher'));assert q==thresholds[n]==float(after['threshold'])
        out={'feature_sequences_checked':len(names),'normal_models_identical_except_threshold':True,'calibration_threshold_reconstructed':True,'config_matches_pre_evaluation_freeze':True,'artifact_validation_only':True}
        Path(f'results/experiment{n}/validation.json').write_text(json.dumps(out,indent=2)+'\n')
    for key in ('visual','process','phases','sequence'):np.testing.assert_array_equal(samples[ns[1]][key],samples[ns[2]][key])
    files=sorted((baseline/'predictions').glob('*.npz'))
    for n in ns[1:]:assert [p.name for p in files]==sorted(p.name for p in (roots[n]/'predictions').glob('*.npz'))
    for path in files:
        predictions={n:load(roots[n]/'predictions'/path.name) for n in ns}
        old=predictions[ns[0]]
        for n,d in predictions.items():
            for key in old:
                if key!='combined':np.testing.assert_array_equal(old[key],d[key],err_msg=n+':'+path.name+':'+key)
            mode=configs[n].get('score_fusion','mean')
            expected=np.maximum(d['visual'],d['process']) if mode=='max' else d['visual'] if mode=='visual' else .5*d['visual']+.5*d['process']
            np.testing.assert_array_equal(d['combined'],expected)
            assert np.isfinite(expected).all();all_predictions[n].append(d)
    joined={n:{key:np.concatenate([p[key] for p in predictions]) for key in ('labels','visual','process','combined')} for n,predictions in all_predictions.items()}
    y=joined[ns[0]]['labels'];max_alarm=joined[ns[1]]['combined']>thresholds[ns[1]];visual_alarm=joined[ns[2]]['combined']>thresholds[ns[2]]
    def counts(mask):return {'normal':int((mask&(y==0)).sum()),'anomaly':int((mask&(y==1)).sum())}
    tails={}
    for n in ns[1:]:
        cal=samples[n]['combined'];test=joined[n]['combined'];q=thresholds[n]
        tails[n]={'normal_calibration_samples':len(cal),'normal_calibration_max':float(cal.max()),'normal_calibration_at_q99':int((cal==q).sum()),'normal_calibration_above_q99':int((cal>q).sum()),'normal_calibration_at_one':int((cal==1).sum()),'test_at_one':counts(test==1),'test_at_q99':counts(test==q),'test_unique_scores':len(np.unique(test)), 'test_process_exceeds_visual':int((joined[n]['process']>joined[n]['visual']).sum())}
        p=Path(f'results/experiment{n}/validation.json');v=json.loads(p.read_text());v.update(test_predictions_checked=len(files),all_branch_object_phase_label_values_identical=True,final_fusion_reconstructed=True);p.write_text(json.dumps(v,indent=2)+'\n')
    result={'scene':scene,'experiments':ns,'thresholds':thresholds,'max_vs_visual_alarm_changes':{'added_by_max':counts(max_alarm&~visual_alarm),'lost_by_max':counts(~max_alarm&visual_alarm),'both':counts(max_alarm&visual_alarm),'neither':counts(~max_alarm&~visual_alarm)},'score_tails':tails,'note':'Each mode has its own normal q99. Diagnostic counts do not tune scores, parameters or thresholds. Same development scene.'}
    out=Path('results/comparison'+'_'.join(ns));out.mkdir(parents=True,exist_ok=True);(out/'fusion_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
