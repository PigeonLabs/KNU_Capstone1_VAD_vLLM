"""Shared immutable inputs and access guards for experiment45."""
import json,sys
from pathlib import Path
from prepare_representation40 import sha,write

OUT=Path('results/experiment45');ART=Path('artifacts/experiment45')
CFG=Path('configs/experiment45_candidates.json')


def config():return json.loads(CFG.read_text())
def verify_record(path):
    record=json.loads(Path(path).read_text())
    for p,h in record['file_sha256'].items():assert sha(p)==h,p
    return record


def guard(normal=True):
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        s=args[0].decode() if isinstance(args[0],bytes) else args[0]
        if '/test_label/' in s:raise RuntimeError('Label access forbidden in this stage')
        if normal and ('/testing/frames/' in s or Path(s).name.startswith('testing_')):
            raise RuntimeError('Test access forbidden in normal stage')
    sys.addaudithook(hook)


def freeze(path,files,**metadata):
    record={**metadata,'file_sha256':{str(p):sha(p) for p in files}}
    if Path(path).exists():assert json.loads(Path(path).read_text())==record,str(path)
    else:write(path,record)
    return record
