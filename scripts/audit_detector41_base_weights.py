"""Verify reused backbone files against their pre-experiment40 hashes."""
import json
from pathlib import Path
from ipad_vad.learned_detector import sha,write

def main():
    old=json.loads(Path('results/experiment40/pre_normal_models_protocol.json').read_text())['file_sha256'];manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text());paths=json.loads(Path('artifacts/model_paths.json').read_text());checked={}
    roots=[Path(manifest['model']['local_path']),Path(paths['openai/clip-vit-base-patch32']['local_path'])]
    for name,h in old.items():
        p=Path(name)
        if any(p.is_relative_to(root) for root in roots):
            assert sha(p)==h,p;checked[name]=h
    assert any(p.endswith('.safetensors') for p in checked) and len(checked)>3
    for name,h in manifest['model']['sha256'].items():
        p=Path(manifest['model']['local_path'])/name;assert sha(p)==h;checked[str(p)]=h
    write('results/experiment41/reused_backbone_integrity.json',{'original_source':'results/experiment40/pre_normal_models_protocol.json','original_protocol_sha256':sha('results/experiment40/pre_normal_models_protocol.json'),'all_checked_files_unchanged':True,'file_sha256':checked})
    print('Verified reused CLIP/MobileCLIP backbone files:',len(checked))

if __name__=='__main__':main()
