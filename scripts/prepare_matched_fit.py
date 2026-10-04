"""Freeze experiment22 normal protocol or its pre-test checkpoint; never overwrite."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from audit_matched_fit import VARIANTS,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--pre-test',action='store_true');args=p.parse_args();out=Path('results/experiment22');out.mkdir(parents=True,exist_ok=True);source=Path('artifacts/experiment19/features/R04')
    for e in ['22',*VARIANTS[2:]]:
        root=Path(f'artifacts/experiment{e}');root.mkdir(parents=True,exist_ok=True);Path(f'results/experiment{e}').mkdir(parents=True,exist_ok=True);link=root/'features'
        if not link.exists():link.symlink_to('../experiment19/features',target_is_directory=True)
        assert (link/'R04').resolve()==source.resolve()
    if args.pre_test:
        audit=json.loads((out/'normal_audit.json').read_text());assert all(v['alarm_not_structurally_blocked'] for v in audit['variants'].values())
        files=[out/'pre_normal_protocol.json',out/'normal_audit.json',out/'normal_holdout.json',Path('scripts/validate_matched_fit.py'),*sorted(Path('artifacts/experiment22/full_normal').glob('*.npz')),*sorted(Path('artifacts/experiment22/normal_holdout').glob('*.npz'))];name='pre_test_checkpoint.json';decision='All seven full calibrations feasible. Evaluate all five fixed random controls; no seed selection or threshold changes.'
    else:
        files=[*[Path(f'configs/experiment{e}.json') for e in VARIANTS],Path('results/experiment18/process_discovery.json'),Path('results/stage00/splits.json'),Path('docs/EXPERIMENT22_PLAN.md'),*sorted(Path('src/ipad_vad').glob('*.py')),*[Path('scripts')/s for s in ['prepare_matched_fit.py','audit_matched_fit.py','audit_factorial_normal.py','evaluate_route_holdout.py','evaluate_baseline.py']]];name='pre_normal_protocol.json';decision='B off. All FIT, observed FIT, count-matched random FIT seeds 0 through 4. No seed selection. Relation masks determine random sample counts only.'
    path=out/name
    if path.exists():
        record=json.loads(path.read_text())
        for p,h in record['file_sha256'].items():assert sha(p)==h,p
        for p,h in record.get('source_features_sha256',{}).items():assert sha(source/p)==h,p
    else:
        record={'created_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'before new test scores' if args.pre_test else 'before normal model fitting','file_sha256':{str(p):sha(p) for p in files},'decision':decision}
        if not args.pre_test:record['source_features_sha256']={p.name:sha(p) for p in sorted(source.glob('*.npz'))}
        path.write_text(json.dumps(record,indent=2)+'\n')
    print(f'Frozen {path}')


if __name__=='__main__':main()
