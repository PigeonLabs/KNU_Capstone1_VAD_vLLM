"""Post-selection test extraction, score freeze, evaluation and validation."""
import json,subprocess,sys,time
from pathlib import Path
from candidates45_common import OUT,ART,config,sha,write,freeze,verify_record
from experiment45_normal import verify_normal


def command(args,stage):
    path=ART/'logs'/f'{stage}.log';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('Existing stage log; inspect terminal state before restart: '+str(path))
    started=time.time();write(OUT/'test_runner_status.json',{'stage':stage,'state':'running','started_unix':started})
    print('START',stage,flush=True)
    with path.open('w') as f:result=subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT)
    write(OUT/'test_runner_status.json',{'stage':stage,'state':'completed' if result.returncode==0 else 'failed',
        'started_unix':started,'elapsed_seconds':time.time()-started,'returncode':result.returncode})
    if result.returncode:raise RuntimeError(f'{stage} failed with {result.returncode}; inspect {path}')
    print('DONE',stage,flush=True)


def main():
    verify_normal();audit=json.loads((OUT/'normal_models_audit.json').read_text())
    assert len(audit['full'])==32 and len(audit['folds'])==176
    assert all(r['finite_bounded'] for r in audit['full']+audit['folds'])
    # Ineligible alarm thresholds are reported, never silently tuned using test labels.
    freeze(OUT/'test_execution_protocol.json',[Path(__file__),OUT/'normal_models_checkpoint.json',OUT/'selection_checkpoint.json',
        Path('scripts/evaluate_candidates45.py'),Path('scripts/validate_candidates45.py'),Path('scripts/report_candidates45.py')],
        selection_frozen_before_test=True,labels_opened=False)
    for name in config()['candidates']:
        complete=OUT/f'{name}_test_extraction.json'
        if not complete.exists():command(['scripts/extract_candidates45.py','--candidate',name,'--partition','test'],name+'_test')
        else:
            verify_record(OUT/f'{name}_test_extraction_protocol.json')
            for row in json.loads(complete.read_text())['sequences']:
                assert sha(ART/name/'features'/row['scene']/f'testing_{row["sequence"]}.npz')==row['sha256']
    command(['scripts/evaluate_candidates45.py'],'evaluate')
    command(['scripts/validate_candidates45.py'],'validate')
    command(['scripts/report_candidates45.py'],'report')
    print('Test evaluation and independent validation completed; inspect figures and write interpretation before publication',flush=True)


if __name__=='__main__':main()
