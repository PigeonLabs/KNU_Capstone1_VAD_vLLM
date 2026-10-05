"""Verify frozen normal baselines and freeze the representation training protocol."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from prepare_representation40 import OUT,sha,write


def main():
    dest=OUT/'representation_protocol.json'
    if dest.exists():
        for p,h in json.loads(dest.read_text())['file_sha256'].items():assert sha(p)==h,p
        print('Existing training protocol verified');return
    manifest=json.loads((OUT/'data_manifest.json').read_text());cfg=json.loads(Path('configs/experiment40_representation.json').read_text());files=[];count=0
    for arm in ['A','B']:
        result=json.loads((OUT/f'{arm}_normal_extraction.json').read_text());assert result['normal_only'] and result['views']==manifest['view_count'] and len(result['sequences'])==111
        for row in result['sequences']:
            p=Path(f'artifacts/experiment40/{arm}/features')/row['scene']/f'training_{row["sequence"]}.npz';source=Path(cfg['source_features'][row['scene']])/p.name
            assert sha(p)==row['sha256'] and sha(source)==manifest['normal_source_feature_sha256'][str(source)]
            with np.load(p,allow_pickle=False) as archive:a=dict(archive)
            with np.load(source,allow_pickle=False) as archive:b=dict(archive)
            assert set(a)==set(b)
            for key in a:
                if key in ['global_features','crop_features']:
                    assert a[key].shape==b[key].shape and np.isfinite(a[key]).all()
                    np.testing.assert_allclose(np.linalg.norm(a[key],axis=-1),1,atol=1e-6)
                else:np.testing.assert_array_equal(a[key],b[key],err_msg=f'{p}:{key}')
            files.append(p);count+=1
    scripts=['prepare_representation40.py','extract_representation40.py','train_representation40.py','run_representation40_training.py','smoke_representation40.py','freeze_representation40.py','extract_features.py']
    files+=[Path('scripts')/s for s in scripts]+list(Path('src/ipad_vad').glob('*.py'))+[Path('tests/test_learned_visual.py'),Path('docs/EXPERIMENT40_PLAN.md'),Path('configs/experiment40_representation.json'),Path('requirements-experiment40.txt'),OUT/'data_manifest.json',OUT/'gpu_smoke.json',OUT/'A_normal_extraction.json',OUT/'B_normal_extraction.json',Path('results/experiment40_R02_base/process_discovery.json')]
    write(dest,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'frozen_before_first_encoder_optimization','normal_feature_files_checked':count,'same_detector_tracks_boxes_phases_and_source_frames':True,'normal_only':True,'file_sha256':{str(p):sha(p) for p in files}})
    print('Frozen training protocol; exact non-visual array checks:',count,flush=True)

if __name__=='__main__':main()
