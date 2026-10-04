"""Freeze experiment27 before normal evaluation and before new test scores."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from audit_transition_gate import VARIANTS,AFTER,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--pre-test',action='store_true');a=p.parse_args();out=Path('results/experiment27');out.mkdir(parents=True,exist_ok=True);source=Path('artifacts/experiment19/features/R04')
    for e in ['27',*AFTER]:
        root=Path(f'artifacts/experiment{e}');root.mkdir(parents=True,exist_ok=True);Path(f'results/experiment{e}').mkdir(parents=True,exist_ok=True);link=root/'features'
        if not link.exists():link.symlink_to('../experiment19/features',target_is_directory=True)
        assert (link/'R04').resolve()==source.resolve()
    if a.pre_test:
        audit=json.loads((out/'normal_audit.json').read_text());assert all(v['alarm_not_structurally_blocked'] for v in audit['variants'].values());h=json.loads((out/'normal_holdout.json').read_text());assert all(r['calibration_finite'] and r['calibration_unit_interval'] and r['normal_q99']<1 for r in h['folds'])
        files=[out/'pre_normal_protocol.json',out/'normal_audit.json',out/'normal_holdout.json',Path('scripts/validate_transition_gate.py'),*sorted(Path('artifacts/experiment27/full_normal').glob('*.npz')),*sorted(Path('artifacts/experiment27/normal_holdout').glob('*.npz'))];name='pre_test_checkpoint.json'
    else:
        files=[*[Path(f'configs/experiment{e}.json') for e in VARIANTS],Path('artifacts/experiment26_hold/normal_model.npz'),Path('results/experiment18/process_discovery.json'),Path('results/stage00/splits.json'),Path('docs/EXPERIMENT27_PLAN.md'),*sorted(Path('src/ipad_vad').glob('*.py')),*[Path('scripts')/s for s in ['prepare_transition_gate.py','audit_transition_gate.py','audit_bank_dispatch.py','audit_request_calibration.py','evaluate_route_holdout.py','evaluate_baseline.py']]];name='pre_normal_protocol.json'
    path=out/name
    if path.exists():
        record=json.loads(path.read_text())
        for p,h in record['file_sha256'].items():assert sha(p)==h,p
    else:
        record={'created_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'before new test scores' if a.pre_test else 'before normal fitting/calibration','file_sha256':{str(p):sha(p) for p in files},'source_features_sha256':{p.name:sha(p) for p in sorted(source.glob('*.npz'))},'decision':'Fixed consecutive-observed transition gate after unchanged calibration. Preserve all three appearance gates and evaluate all feasible candidates; own normal q99, no label-based choice.'};path.write_text(json.dumps(record,indent=2)+'\n')
    print(path)


if __name__=='__main__':main()
