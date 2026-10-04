"""Freeze experiment21 before normal model fitting and evaluation."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=Path('results/experiment21');out.mkdir(parents=True,exist_ok=True);source=Path('artifacts/experiment19/features/R04')
    for e in ['21','21_fit','21_infer']:
        root=Path(f'artifacts/experiment{e}');root.mkdir(parents=True,exist_ok=True);Path(f'results/experiment{e}').mkdir(parents=True,exist_ok=True);link=root/'features'
        if not link.exists():link.symlink_to('../experiment19/features',target_is_directory=True)
        assert (link/'R04').resolve()==source.resolve()
    files=[*[Path(f'configs/experiment{e}.json') for e in ['18','19','21_fit','21_infer']],Path('results/experiment18/process_discovery.json'),Path('results/stage00/splits.json'),Path('docs/EXPERIMENT21_PLAN.md'),*sorted(Path('src/ipad_vad').glob('*.py')),*[Path('scripts')/p for p in ['prepare_factorial_appearance.py','audit_factorial_normal.py','evaluate_route_holdout.py','check_normal_feasibility.py','evaluate_baseline.py']]]
    path=out/'pre_normal_protocol.json'
    if path.exists():
        record=json.loads(path.read_text())
        for p,h in record['file_sha256'].items():assert sha(p)==h,p
        for p,h in record['source_features_sha256'].items():assert sha(source/p)==h,p
    else:path.write_text(json.dumps({'created_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'Before all experiment21 normal holdout, full calibration and new test scores','file_sha256':{str(p):sha(p) for p in files},'source_features_sha256':{p.name:sha(p) for p in sorted(source.glob('*.npz'))},'decision':'Prespecified 2x2 A=observed FIT restriction, B=missing relation pooled inference. Role-wide CDF in all four cells. Publish every cell; no tuning or test-based selection.'},indent=2)+'\n')
    print('Frozen experiment21 factorial protocol; all features reused unchanged.')


if __name__=='__main__':main()
