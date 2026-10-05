"""Download the pinned official OpenCLIP conversion; no remote Python code."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

REPO='timm/MobileCLIP2-S2-OpenCLIP'
REVISION='ac6b37c8fc40b62b623d09ed228a6f0c9bc29fe6'


def main():
    path=snapshot_download(REPO,revision=REVISION,cache_dir='.cache/huggingface/hub',allow_patterns=['open_clip_config.json','open_clip_model.safetensors','README.md'])
    record={'repo':REPO,'revision':REVISION,'local_path':str(Path(path))};out=Path('artifacts/experiment40/mobileclip_model.json');out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        old=json.loads(out.read_text());assert old['repo']==REPO and old['revision']==REVISION and Path(old['local_path']).resolve()==Path(path).resolve()
    else:out.write_text(json.dumps(record,indent=2)+'\n')
    print(out)

if __name__=='__main__':main()
