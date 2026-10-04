"""Freeze experiment20 inputs before normal holdout and test scoring."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=Path('results/experiment20');out.mkdir(parents=True,exist_ok=True);art=Path('artifacts/experiment20');art.mkdir(parents=True,exist_ok=True)
    source=Path('artifacts/experiment19/features/R04');link=art/'features'
    if not link.exists():link.symlink_to('../experiment19/features',target_is_directory=True)
    assert (link/'R04').resolve()==source.resolve()
    files=[Path('configs/experiment20.json'),Path('configs/experiment19.json'),Path('results/experiment18/process_discovery.json'),Path('results/stage00/splits.json'),Path('docs/EXPERIMENT20_PLAN.md'),*sorted(Path('src/ipad_vad').glob('*.py')),*[Path('scripts')/p for p in ['prepare_route_calibration.py','evaluate_route_holdout.py','check_normal_feasibility.py','evaluate_baseline.py']]]
    path=out/'pre_normal_protocol.json'
    if path.exists():
        saved=json.loads(path.read_text())
        for p,h in saved['file_sha256'].items():assert sha(p)==h,p
        for p,h in saved['source_features_sha256'].items():assert sha(source/p)==h,p
    else:path.write_text(json.dumps({'created_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'Before experiment20 normal holdout, full calibration and test','file_sha256':{str(p):sha(p) for p in files},'source_features_sha256':{p.name:sha(p) for p in sorted(source.glob('*.npz'))},'decision':'Fixed role x actual-bank-route CDF; minimum 50 observations and 2 videos, otherwise role-wide fallback. No selection by holdout or test scores.'},indent=2)+'\n')
    print('Frozen experiment20 protocol; feature arrays reused byte-for-byte.')


if __name__=='__main__':main()
