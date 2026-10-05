"""Verify the normal-only runtime recovery did not alter supported processes."""
import json
from pathlib import Path
import numpy as np
from ipad_vad.learned_detector import sha,write

OUT=Path('results/experiment41')

def main():
    record=json.loads((OUT/'normal_fit_recovery.json').read_text())
    for p,h in record['archived_public_files'].items():assert sha(p)==h,p
    original=Path('artifacts/experiment41/normal_fit_attempt1/normal_models');checked=0
    for old in sorted(original.glob('*.npz')):
        assert any(f'_{s}_' in old.name for s in ['R01','R02','R03'])
        with np.load(old,allow_pickle=False) as a,np.load(Path('artifacts/experiment41/normal_models')/old.name,allow_pickle=False) as b:
            assert set(a)==set(b)
            for k in a:np.testing.assert_array_equal(a[k],b[k],err_msg=old.name+':'+k)
        checked+=1
    assert checked==320,checked
    audit=json.loads((OUT/'normal_models_audit.json').read_text());r04=[r for r in audit['full'] if r['scene']=='R04'];assert len(r04)==8
    for r in r04:
        assert r['dwell_available'] is False and r['dwell_context_support']==record['context_support']
        for path in Path('artifacts/experiment41/normal_models').glob(f'{r["run"]}_R04_*_scores.npz'):
            with np.load(path,allow_pickle=False) as f:assert not f['dwell_valid'].any() and not f['dwell'].any()
    old_protocol=json.loads((OUT/'normal_fit_attempt1/pre_normal_models_protocol.json').read_text())['file_sha256'];now=json.loads((OUT/'pre_normal_models_protocol.json').read_text())['file_sha256']
    changed=[p for p,h in old_protocol.items() if now.get(p)!=h]
    expected=['scripts/experiment41_normal.py',*[str(p) for p in Path('configs').glob('experiment41_*_R04.json')]]
    assert {Path(p).resolve() for p in changed}=={Path(p).resolve() for p in expected},(changed,expected)
    write(OUT/'normal_fit_recovery_validation.json',{'archived_normal_model_and_score_files_exactly_reproduced':checked,'unchanged_supported_processes':['R01','R02','R03'],'all8_R04_dwell_unavailable_and_zero_masked':True,'changed_existing_protocol_files':changed,'new_protocol_files':sorted(set(now)-set(old_protocol)),'all_original_training_weights_and_features_unchanged':True,'test_labels_accessed':False})
    print('Recovery validation:',checked,'normal model/score files unchanged')

if __name__=='__main__':main()
