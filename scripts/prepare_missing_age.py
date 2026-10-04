"""Freeze FIT-only missing-gap threshold before calibration, then freeze pre-test evidence."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from ipad_vad.missing_age import MissingAge
from evaluate_route_holdout import load_cache
from audit_missing_age import VARIANTS,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--pre-test',action='store_true');args=p.parse_args();out=Path('results/experiment24');out.mkdir(parents=True,exist_ok=True);source=Path('artifacts/experiment19/features/R04')
    for e in ['24',*VARIANTS[1:]]:
        root=Path(f'artifacts/experiment{e}');root.mkdir(parents=True,exist_ok=True);Path(f'results/experiment{e}').mkdir(parents=True,exist_ok=True);link=root/'features'
        if not link.exists():link.symlink_to('../experiment19/features',target_is_directory=True)
        assert (link/'R04').resolve()==source.resolve()
    if args.pre_test:
        audit=json.loads((out/'normal_audit.json').read_text());assert all(v['alarm_not_structurally_blocked'] for v in audit['variants'].values())
        files=[out/'pre_normal_protocol.json',out/'normal_audit.json',out/'normal_holdout.json',Path('scripts/validate_missing_age.py'),*sorted(Path('artifacts/experiment24/full_normal').glob('*.npz')),*sorted(Path('artifacts/experiment24/normal_holdout').glob('*.npz'))];name='pre_test_checkpoint.json';decision='Both prespecified candidates feasible; evaluate both, no tau or threshold tuning.'
    else:
        split=json.loads(Path('results/stage00/splits.json').read_text())['R04'];cfg=json.loads(Path('configs/experiment24_age.json').read_text());m=MissingAge(**cfg['appearance_missing_age']);m.fit([load_cache(source,s) for s in split['fit']]);profile=out/'fit_gap_profile.json'
        if profile.exists():assert json.loads(profile.read_text())==m.report()
        else:profile.write_text(json.dumps(m.report(),indent=2)+'\n')
        files=[profile,*[Path(f'configs/experiment{e}.json') for e in VARIANTS],Path('results/experiment18/process_discovery.json'),Path('results/stage00/splits.json'),Path('docs/EXPERIMENT24_PLAN.md'),*sorted(Path('src/ipad_vad').glob('*.py')),*[Path('scripts')/s for s in ['prepare_missing_age.py','audit_missing_age.py','audit_factorial_normal.py','evaluate_route_holdout.py','evaluate_baseline.py']]];name='pre_normal_protocol.json';decision='FIT-only complete-gap 90th percentile fixed before calibration. Compare unlimited hold, immediate pooled and causal age-limited hold. No threshold sweep.'
    path=out/name
    if path.exists():
        record=json.loads(path.read_text())
        for p,h in record['file_sha256'].items():assert sha(p)==h,p
        for p,h in record.get('source_features_sha256',{}).items():assert sha(source/p)==h,p
    else:
        record={'created_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'before new test scores' if args.pre_test else 'FIT gap profile fixed; before normal calibration','file_sha256':{str(p):sha(p) for p in files},'decision':decision}
        if not args.pre_test:record['source_features_sha256']={p.name:sha(p) for p in sorted(source.glob('*.npz'))}
        path.write_text(json.dumps(record,indent=2)+'\n')
    print(f'Frozen {path}')


if __name__=='__main__':main()
