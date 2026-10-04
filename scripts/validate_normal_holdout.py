"""Reconstruct every normal-only fold reference, prediction and reported rate."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.normal_validation import make_model,fit_arrays,array_fingerprint,reference_arrays
from ipad_vad.experiment import load_process
from ipad_vad.data import hold_scores


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    out=Path('results/experiment13');cfg=json.loads(Path('configs/experiment13.json').read_text());record=json.loads((out/'pre_evaluation_protocol.json').read_text())
    assert hashlib.sha256(Path('configs/experiment13.json').read_bytes()).hexdigest()==record['config_sha256']
    assert hashlib.sha256((out/'provenance.json').read_bytes()).hexdigest()==record['provenance_sha256']
    for path,sha in record['code_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    provenance=json.loads((out/'provenance.json').read_text())
    for path,sha in provenance['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    metrics=json.loads((out/'metrics.json').read_text());split=json.loads(Path(cfg['split_source']).read_text())[cfg['scene']]
    source=Path(f'artifacts/experiment{cfg["source_features_experiment"]}/features/{cfg["scene"]}')
    caches={s:load(source/f'training_{s}.npz') for s in split['fit']+split['calibration']};count=0;cross={}
    with threadpool_limits(limits=4):
        for variant in cfg['variants']:
            assert hashlib.sha256(Path(f'configs/experiment{variant}.json').read_bytes()).hexdigest()==record['variant_config_sha256'][variant]
            vc=json.loads(Path(f'configs/experiment{variant}.json').read_text());model=make_model(vc,load_process(vc));model.fit([caches[s] for s in split['fit']])
            fingerprint=array_fingerprint(fit_arrays(model));assert fingerprint==metrics['fit_checks'][variant]['fit_fingerprint']
            for fold in record['folds']:
                held=fold['held_out'];ids=fold['calibration'];assert held not in ids and held not in split['fit'] and set(ids)==set(split['calibration'])-{held}
                m=deepcopy(model);m.calibrate([caches[s] for s in ids]);root=Path(f'artifacts/experiment13/variant{variant}/holdout_{held}')
                saved_model=load(root/'normal_model.npz');saved_cal=load(root/'calibration_scores.npz');saved=load(root/'heldout_scores.npz')
                for key,value in {**fit_arrays(m),**reference_arrays(m)}.items():np.testing.assert_array_equal(saved_model[key],value,err_msg=variant+':'+held+':'+key)
                assert set(saved_cal['sequence'].tolist())==set(ids) and set(saved['sequence'].tolist())=={held}
                for branch in ['visual','process','combined']:
                    cal=np.concatenate([m.score(caches[s])[branch] for s in ids]);np.testing.assert_array_equal(saved_cal[branch],cal)
                    score=m.score(caches[held])[branch];np.testing.assert_array_equal(saved[branch],score)
                    np.testing.assert_array_equal(saved['dense_'+branch],hold_scores(caches[held]['indices'],score,int(caches[held]['frame_count'])))
                assert m.threshold==np.quantile(saved_cal['combined'],vc['calibration_quantile'],method='higher')
                alarm=saved['dense_combined']>m.threshold;np.testing.assert_array_equal(saved['dense_alarm'],alarm)
                row=next(r for r in metrics['folds'] if r['variant']==variant and r['held_out']==held)
                assert row['frames']==len(alarm) and row['false_positive_frames']==int(alarm.sum()) and row['frame_fpr']==float(alarm.mean())
                assert array_fingerprint(fit_arrays(m))==fingerprint
                cross[(variant,held)]=(saved,saved_cal);count+=1
    for held in split['calibration']:
        baseline,baseline_cal=cross[('10',held)]
        for variant in cfg['variants'][1:]:
            actual,cal=cross[(variant,held)]
            np.testing.assert_array_equal(baseline['visual'],actual['visual']);np.testing.assert_array_equal(baseline['process'],actual['transition'])
            np.testing.assert_array_equal(baseline_cal['visual'],cal['visual']);np.testing.assert_array_equal(baseline_cal['process'],cal['transition'])
        for part in [0,1]:np.testing.assert_array_equal(cross[('12',held)][part]['dwell_valid'],cross[('12_support',held)][part]['dwell_valid'])
    for variant,summary in metrics['summaries'].items():
        rows=[r for r in metrics['folds'] if r['variant']==variant]
        assert summary['micro_frame_fpr']==sum(r['false_positive_frames'] for r in rows)/sum(r['frames'] for r in rows)
        assert summary['macro_video_frame_fpr']==float(np.mean([r['frame_fpr'] for r in rows]))
    result={'folds_reconstructed':count,'normal_feature_videos':len(caches),'fit_and_all_calibration_references_exclude_heldout':True,'all_scores_q99_alarm_counts_and_aggregates_reconstructed':True,
        'fixed_fit_fingerprints_preserved':True,'common_visual_transition_branches_identical_between_variants':True,'context_arms_have_identical_masks':True,
        'configuration_source_and_code_hashes_match':True,'artifact_validation_only':True,'limitation':'Normal-only evidence; no anomaly accuracy or original recording-group independence established.'}
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
