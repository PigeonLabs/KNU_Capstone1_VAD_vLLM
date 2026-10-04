"""Download the pinned public backbones; no access tokens or paid APIs required."""
import json
import os
from pathlib import Path
os.environ.setdefault('HF_HOME',str(Path('.cache/huggingface').resolve()))
from huggingface_hub import snapshot_download

revisions=json.loads(Path('results/model_revisions.json').read_text())
paths={}
for name,info in revisions.items():
    path=snapshot_download(name,revision=info['revision'],allow_patterns=['*.json','*.txt','*.safetensors','pytorch_model.bin'])
    paths[name]={'revision':info['revision'],'local_path':path}
Path('artifacts').mkdir(exist_ok=True)
Path('artifacts/model_paths.json').write_text(json.dumps(paths,indent=2)+'\n')
