"""Freeze actual-bank CDF dispatch experiment26."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from audit_bank_dispatch import VARIANTS,AFTER,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--pre-test',action='store_true');args=p.parse_args();out=Path('results/experiment26');out.mkdir(parents=True,exist_ok=True);source=Path('artifacts/experiment19/features/R04')
    for e in ['26',*AFTER]:
        root=Path(f'artifacts/experiment{e}');root.mkdir(parents=True,exist_ok=True);Path(f'results/experiment{e}').mkdir(parents=True,exist_ok=True);link=root/'features'
        if not link.exists():link.symlink_to('../experiment19/features',target_is_directory=True)
        assert (link/'R04').resolve()==source.resolve()
    if args.pre_test:
        audit=json.loads((out/'normal_audit.json').read_text());assert all(v['alarm_not_structurally_blocked'] for v in audit['variants'].values())
        files=[out/'pre_normal_protocol.json',out/'normal_audit.json',out/'normal_holdout.json',Path('scripts/validate_bank_dispatch.py'),*sorted(Path('artifacts/experiment26/full_normal').glob('*.npz')),*sorted(Path('artifacts/experiment26/normal_holdout').glob('*.npz'))];name='pre_test_checkpoint.json';decision='All three fixed candidates feasible; evaluate every candidate. No gate or threshold selection.'
    else:
        files=[*[Path(f'configs/experiment{e}.json') for e in VARIANTS],Path('results/experiment24/fit_gap_profile.json'),Path('artifacts/experiment25_hold/normal_model.npz'),Path('results/experiment18/process_discovery.json'),Path('results/stage00/splits.json'),Path('docs/EXPERIMENT26_PLAN.md'),*sorted(Path('src/ipad_vad').glob('*.py')),*[Path('scripts')/s for s in ['prepare_bank_dispatch.py','audit_bank_dispatch.py','audit_missing_age.py','audit_factorial_normal.py','audit_request_calibration.py','evaluate_route_holdout.py','evaluate_baseline.py']]];name='pre_normal_protocol.json';decision='Keep both full-normal references fixed. Dispatch by actual subspace family, using canonical pooled residual for support fallback. Fixed FIT/PCA/process/gates/tau; own q99. No gate or parameter selection.'
    path=out/name
    if path.exists():
        record=json.loads(path.read_text())
        for p,h in record['file_sha256'].items():assert sha(p)==h,p
        for p,h in record.get('source_features_sha256',{}).items():assert sha(source/p)==h,p
    else:
        record={'created_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'before new test scores' if args.pre_test else 'before normal model fitting/calibration','file_sha256':{str(p):sha(p) for p in files},'decision':decision}
        if not args.pre_test:record['source_features_sha256']={p.name:sha(p) for p in sorted(source.glob('*.npz'))}
        path.write_text(json.dumps(record,indent=2)+'\n')
    print(f'Frozen {path}')


if __name__=='__main__':main()
