"""Encode the precommitted normal role descriptions with the existing frozen CLIP."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from transformers import CLIPModel,CLIPProcessor


def main():
    cfg=json.loads(Path('configs/experiment17.json').read_text());protocol=json.loads(Path('results/experiment17/pre_margin_protocol.json').read_text())
    assert hashlib.sha256(Path('configs/experiment17.json').read_bytes()).hexdigest()==protocol['config_sha256']
    source=json.loads(Path('artifacts/model_paths.json').read_text())[cfg['encoder']];options=cfg['anchor_verification'];prompts=[options['positive'],options['negative']]
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent model/device substitution')
    torch.manual_seed(cfg['seed']);torch.set_num_threads(4)
    model=CLIPModel.from_pretrained(source['local_path'],local_files_only=True).eval().cuda()
    for p in model.parameters():p.requires_grad_(False)
    processor=CLIPProcessor.from_pretrained(source['local_path'],local_files_only=True)
    inputs=processor(text=prompts,padding=True,return_tensors='pt').to('cuda')
    with torch.inference_mode():
        features=model.get_text_features(**inputs);features=features/features.norm(dim=-1,keepdim=True)
    path=Path(options['text_features']);np.savez_compressed(path,text_features=features.cpu().numpy())
    result={'model':cfg['encoder'],'revision':source['revision'],'prompts':prompts,'margin_threshold':options['margin_threshold'],'text_features_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dimension':features.shape[1],'device':'cuda','frozen':True,'network_used':False,'human_normal_semantic_input':True}
    Path('results/experiment17/text_encoding.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
