"""Sequential GPU work only; failed stages stop and retain completed artifacts."""
import json,subprocess,sys,time
from pathlib import Path
from candidates45_common import OUT,ART,config,sha,write,verify_record


def command(args,stage):
    path=ART/'logs'/f'{stage}.log';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('Existing stage log; inspect terminal status before any restart: '+str(path))
    started=time.time();write(OUT/'normal_runner_status.json',{'stage':stage,'state':'running','started_unix':started})
    print('START',stage,flush=True)
    with path.open('w') as f:result=subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT)
    write(OUT/'normal_runner_status.json',{'stage':stage,'state':'completed' if result.returncode==0 else 'failed',
        'started_unix':started,'elapsed_seconds':time.time()-started,'returncode':result.returncode})
    if result.returncode:raise RuntimeError(f'{stage} failed with {result.returncode}; inspect {path}')
    print('DONE',stage,flush=True)


def main():
    for name in config()['candidates']:
        # Completion records are checked, not used as evidence of a live process.
        if not (OUT/f'{name}_smoke.json').exists():
            command(['scripts/extract_candidates45.py','--candidate',name,'--smoke'],name+'_smoke')
        if not (OUT/f'{name}_normal_extraction.json').exists():
            command(['scripts/extract_candidates45.py','--candidate',name],name+'_normal')
        else:
            verify_record(OUT/f'{name}_normal_extraction_protocol.json')
            record=json.loads((OUT/f'{name}_normal_extraction.json').read_text())
            for row in record['sequences']:
                assert sha(ART/name/'features'/row['scene']/f'training_{row["sequence"]}.npz')==row['sha256']
        if not (OUT/f'{name}_probe.json').exists():command(['scripts/probe_candidates45.py','--candidate',name],name+'_probe')
    command(['scripts/probe_candidates45.py','--select'],'selection')
    command(['scripts/experiment45_normal.py'],'normal_models')
    print('Normal candidate stage completed; inspect selection and normal audit before test stage',flush=True)


if __name__=='__main__':main()
