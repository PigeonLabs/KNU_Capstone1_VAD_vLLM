"""Complete the frozen experiment41 stages sequentially; stop at any failed check."""
import json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/experiment41';ART=ROOT/'artifacts/experiment41';PY=str(ROOT/'.venv/bin/python')
RUNS=['A','B','C_s42','D_s42','C_s43','D_s43','C_s44','D_s44']


def stage(name,args,completed):
    if completed.exists():print('Existing stage:',name,flush=True);return
    with (ART/f'{name}.log').open('w') as f:
        result=subprocess.run([PY,*args],cwd=ROOT,env={**os.environ,'PYTHONPATH':'artifacts/experiment40/deps:src:scripts'},stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed; inspect {ART/name}.log')
    if not completed.exists():raise RuntimeError(f'{name} did not write completion record')
    print('Completed:',name,flush=True)


def main():
    # An extraction already launched in this task owns the GPU until its final record.
    deadline=time.monotonic()+7200
    while not (OUT/'detector_training_extraction.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Normal detector extraction did not finish; inspect its job')
        time.sleep(5)
    for run in RUNS:stage(f'{run}_normal',['scripts/extract_detector41_representations.py','--run',run,'--partition','training'],OUT/f'{run}_training_extraction.json')
    stage('normal_models',['scripts/experiment41_normal.py'],OUT/'normal_models_checkpoint.json')
    stage('normal_diagnostics',['scripts/diagnose_detector41.py'],OUT/'normal_diagnostics.json')
    stage('detector_test',['scripts/extract_detector41.py','--partition','testing'],OUT/'detector_testing_extraction.json')
    stage('test_manifest',['scripts/prepare_detector41_test.py'],OUT/'test_data_manifest.json')
    for run in RUNS:stage(f'{run}_test',['scripts/extract_detector41_representations.py','--run',run,'--partition','testing'],OUT/f'{run}_testing_extraction.json')
    stage('evaluation',['scripts/evaluate_detector41.py'],OUT/'metrics.json')
    stage('validation',['scripts/validate_detector41.py'],OUT/'validation.json')
    stage('loss_path_audit',['scripts/audit_detector41_loss.py'],OUT/'loss_path_audit.json')
    print('Experiment41 evaluation and validation complete; reporting/publication still required.',flush=True)


if __name__=='__main__':main()
