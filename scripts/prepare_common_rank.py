"""Freeze common-rank experiment23 before normal runs and before new test evaluation."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from audit_common_rank import VARIANTS,BEFORE,AFTER,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pre-test',action='store_true');args=parser.parse_args();out=Path('results/experiment23');out.mkdir(parents=True,exist_ok=True);source=Path('artifacts/experiment19/features/R04')
    for e in ['23',*AFTER]:
        art=Path(f'artifacts/experiment{e}');art.mkdir(parents=True,exist_ok=True);Path(f'results/experiment{e}').mkdir(parents=True,exist_ok=True);link=art/'features'
        if not link.exists():link.symlink_to('../experiment19/features',target_is_directory=True)
        assert (link/'R04').resolve()==source.resolve()
    if args.pre_test:
        audit=json.loads((out/'normal_audit.json').read_text());assert all(v['alarm_not_structurally_blocked'] for v in audit['variants'].values())
        files=[out/'pre_normal_protocol.json',out/'normal_audit.json',out/'normal_holdout.json',Path('scripts/validate_common_rank.py'),*sorted(Path('artifacts/experiment23/full_normal').glob('*.npz')),*sorted(Path('artifacts/experiment23/normal_holdout').glob('*.npz'))];name='pre_test_checkpoint.json';decision='All six fixed common-rank candidates feasible. Evaluate every candidate; no seed or threshold selection.'
    else:
        files=[*[Path(f'configs/experiment{e}.json') for e in VARIANTS],Path('results/experiment22/normal_audit.json'),Path('results/experiment18/process_discovery.json'),Path('results/stage00/splits.json'),Path('docs/EXPERIMENT23_PLAN.md'),*sorted(Path('src/ipad_vad').glob('*.py')),*[Path('scripts')/s for s in ['prepare_common_rank.py','audit_common_rank.py','audit_matched_fit.py','audit_factorial_normal.py','evaluate_route_holdout.py','evaluate_baseline.py']],*[Path(f'artifacts/experiment{e}/normal_model.npz') for e in BEFORE]];name='pre_normal_protocol.json';decision='Six fixed FIT selections from experiment22. Per-phase common rank = minimum of their normal FIT ranks. Keep mean, samples, pooled banks and process unchanged. No seed selection.'
    path=out/name
    if path.exists():
        record=json.loads(path.read_text())
        for p,h in record['file_sha256'].items():assert sha(p)==h,p
        for p,h in record.get('source_features_sha256',{}).items():assert sha(source/p)==h,p
    else:
        record={'created_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'before new test scores' if args.pre_test else 'before normal fitting/calibration','file_sha256':{str(p):sha(p) for p in files},'decision':decision}
        if not args.pre_test:record['source_features_sha256']={p.name:sha(p) for p in sorted(source.glob('*.npz'))}
        path.write_text(json.dumps(record,indent=2)+'\n')
    print(f'Frozen {path}')


if __name__=='__main__':main()
