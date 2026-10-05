"""Pinned official weights; cache stays on the data disk; never execute remote code."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from prepare_representation40 import sha, write


def main():
    cfg=json.loads(Path('configs/experiment45_candidates.json').read_text())
    out=Path('results/experiment45');out.mkdir(parents=True,exist_ok=True)
    records={}
    for name,model in cfg['candidates'].items():
        path=Path(snapshot_download(model['repo'],revision=model['revision'],
            cache_dir='.cache/huggingface/hub',
            allow_patterns=['config.json','preprocessor_config.json','model.safetensors','README.md']))
        records[name]={**model,'local_path':str(path.resolve()),
            'sha256':{p.name:sha(p) for p in sorted(path.iterdir()) if p.name in ['config.json','preprocessor_config.json','model.safetensors','README.md']}}
        assert all((path/n).exists() for n in ['config.json','preprocessor_config.json','model.safetensors'])
        dest=out/f'{name}_model.json'
        if dest.exists():assert json.loads(dest.read_text())==records[name]
        else:write(dest,records[name])
        print(name,model['revision'],'downloaded and hashed',flush=True)


if __name__=='__main__':main()
