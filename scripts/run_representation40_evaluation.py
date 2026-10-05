"""Continue the already-running experiment queue through frozen-model evaluation."""
import json,subprocess,sys,time,shutil
from pathlib import Path
from prepare_representation40 import OUT,write
from experiment40_normal import RUNS


def main():
    status={'status':'waiting_for_training','completed':[],'active':None};target=OUT/'evaluation_queue.json'
    if target.exists():raise RuntimeError('Existing evaluation queue; inspect state instead of duplicate launch')
    write(target,status)
    while True:
        training=json.loads((OUT/'training_queue.json').read_text())
        if training['status']=='failed':raise RuntimeError('Training failed; no automatic recovery')
        if training['status']=='complete':break
        time.sleep(10)
    jobs=[]
    for run in RUNS[2:]:
        arm,seed=run.split('_s');jobs.append((run+'_normal',['scripts/extract_adapted40.py','--arm',arm,'--seed',seed]))
    jobs += [('normal_models',['scripts/experiment40_normal.py']),('R02_test_base',['scripts/extract_features.py','--data-root','/media/jeong/ExtHDD/IPAD_vLLM/IPAD_dataset','--config','configs/experiment40_R02_base.json','--partition','testing']),('test_prepare',['scripts/extract_representation40_test.py','--prepare'])]
    jobs += [(run+'_test',['scripts/extract_representation40_test.py','--run',run]) for run in RUNS]
    jobs += [('scoring_and_metrics',['scripts/evaluate_representation40.py'])]
    for name,args in jobs:
        status.update(status='running',active=name);write(target,status)
        if name=='R02_test_base':
            p=Path('results/experiment40_R02_base/extraction.json');copy=p.with_name('normal_extraction.json')
            if copy.exists():assert copy.read_bytes()==p.read_bytes()
            else:shutil.copyfile(p,copy)
        log=Path('/tmp/experiment40_eval_'+name+'.log');print('Starting',name,flush=True)
        with log.open('w') as f:r=subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:
            status.update(status='failed',returncode=r.returncode,log=str(log));write(target,status);raise RuntimeError(f'{name} failed; inspect {log}')
        status['completed'].append(name);write(target,status)
    status.update(status='complete',active=None);write(target,status);print('All frozen-model evaluations complete',flush=True)

if __name__=='__main__':main()
