"""Run the predeclared C/D seed pairs sequentially, without automatic restarts."""
import json,subprocess,sys,time
from pathlib import Path
from prepare_representation40 import OUT,write


def main():
    cfg=json.loads(Path('configs/experiment40_representation.json').read_text())
    if not (OUT/'A_normal_extraction.json').exists() or not (OUT/'B_normal_extraction.json').exists():raise RuntimeError('Complete frozen normal extractions first')
    if not (OUT/'gpu_smoke.json').exists():raise RuntimeError('Complete GPU smoke first')
    jobs=[(arm,seed) for seed in cfg['seeds'] for arm in ['C','D']]
    status={'jobs':[f'{a}_s{s}' for a,s in jobs],'completed':[],'active':None,'status':'running'}
    for arm,seed in jobs:
        run=f'{arm}_s{seed}';status['active']=run;write(OUT/'training_queue.json',status)
        log=Path(f'/tmp/experiment40_{run}_train.log');print('Starting',run,flush=True)
        with log.open('w') as f:
            result=subprocess.run([sys.executable,'scripts/train_representation40.py','--arm',arm,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(status='failed',failed_run=run,returncode=result.returncode,log=str(log));write(OUT/'training_queue.json',status);raise RuntimeError(f'{run} failed; inspect {log} before deciding recovery')
        status['completed'].append(run)
    status.update(active=None,status='complete');write(OUT/'training_queue.json',status);print('All six adaptation runs complete',flush=True)

if __name__=='__main__':main()
