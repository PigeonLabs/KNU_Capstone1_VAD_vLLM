"""Freeze normal-only provenance audit and protect preexisting model/score artifacts."""
import json
from pathlib import Path
from datetime import datetime,timezone
from audit_missing_age import sha


def main():
    cfg=json.loads(Path('configs/experiment29.json').read_text());out=Path('results/experiment29');out.mkdir(parents=True,exist_ok=True);Path('artifacts/experiment29').mkdir(parents=True,exist_ok=True);split=json.loads(Path('results/stage00/splits.json').read_text())['R04']
    protected=[Path(cfg['feature_source'])/f'training_{s}.npz' for s in split['fit']+split['calibration']]
    for e in ['27','28']:
        for folder in ['full_normal','normal_holdout']:protected.extend(sorted(Path(f'artifacts/experiment{e}/{folder}').glob('*.npz')))
        for variant in ['hold','pool','age']:
            for name in ['normal_model.npz','normal_calibration_scores.npz']:protected.append(Path(f'artifacts/experiment{e}_{variant}/{name}'))
    files=[Path('configs/experiment29.json'),Path(cfg['relation_config']),Path(cfg['relation_model']),Path('artifacts/experiment17/anchor_text_features.npz'),Path('results/stage00/splits.json'),Path('docs/EXPERIMENT29_PLAN.md'),Path('scripts/prepare_transition_provenance.py'),Path('scripts/audit_transition_provenance.py'),Path('src/ipad_vad/transition_trace.py'),*[Path('src/ipad_vad')/name for name in ['relational_phase.py','verified_anchor.py','temporal_anchor.py']]];dest=out/'pre_audit_protocol.json'
    if dest.exists():
        record=json.loads(dest.read_text())
        for p,h in {**record['code_input_sha256'],**record['protected_sha256']}.items():assert sha(p)==h,p
    else:dest.write_text(json.dumps({'created_at_utc':datetime.now(timezone.utc).isoformat(),'normal_only':True,'stage':'before coverage analysis/case selection/image review','code_input_sha256':{str(p):sha(p) for p in files},'protected_sha256':{str(p):sha(p) for p in protected},'test_data_opened':False},indent=2)+'\n')
    print(dest)


if __name__=='__main__':main()
