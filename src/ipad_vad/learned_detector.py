"""Shared GroundingDINO decoder adaptation with reviewed normal weak boxes."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps, build_text_mask


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_detector():
    record = json.loads(Path('artifacts/model_paths.json').read_text())['IDEA-Research/grounding-dino-tiny']
    processor = AutoProcessor.from_pretrained(record['local_path'], local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(record['local_path'], local_files_only=True)
    return model, processor, record


def configure_detector(model):
    model.requires_grad_(False)
    names = []
    for name, p in model.named_parameters():
        if name.startswith(('model.decoder.', 'bbox_embed.')):
            p.requires_grad_(True); names.append(name)
    if not names: raise ValueError('No decoder/head parameters found')
    # Keep pretrained stochastic layers and all frozen buffers fixed for this pilot.
    model.eval()
    return {'trainable_names': names, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_parameters': sum(p.numel() for p in model.parameters()), 'mode': 'eval_with_decoder_and_bbox_gradients'}


def state_hash(model, frozen):
    h = hashlib.sha256()
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        if (not p.requires_grad) != frozen: continue
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def adapted_state(model):
    return {name: p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}


def restore_adapted(model, state):
    expected = {name for name,p in model.named_parameters() if p.requires_grad}
    if set(state) != expected: raise ValueError('Checkpoint adaptation scope differs')
    result = model.load_state_dict(state, strict=False)
    assert not result.unexpected_keys


def scene_batches(rows, batch_size=2, rng=None):
    # HF 4.57.6 loss class offsets require equal role counts for this batch=2 path.
    # Same-scene batches also avoid mixing the prompt-token spaces.
    if batch_size != 2: raise ValueError('Pilot is validated for batch size 2 only')
    batches = []
    for scene in sorted({r['scene'] for r in rows}):
        selected = [r for r in rows if r['scene'] == scene]
        if rng is not None: rng.shuffle(selected)
        batches.extend(selected[i:i+batch_size] for i in range(0,len(selected),batch_size))
    if rng is not None: rng.shuffle(batches)
    return batches


def batch_inputs(rows, processor, cfg, rng=None, device='cuda'):
    if len({r['scene'] for r in rows}) != 1 or len(rows) > 2: raise ValueError('Same-scene batches of at most 2 required')
    prompts = cfg['prompts'][rows[0]['scene']]
    caption = '. '.join(prompts) + '.'
    images=[]; labels=[]
    for r in rows:
        with Image.open(Path(cfg['data_root']) / r['path']) as f: im=f.convert('RGB')
        if rng is not None:
            im=ImageEnhance.Brightness(im).enhance(float(rng.uniform(.9,1.1)))
            im=ImageEnhance.Contrast(im).enhance(float(rng.uniform(.9,1.1)))
            im=ImageEnhance.Color(im).enhance(float(rng.uniform(.95,1.05)))
        images.append(im)
        b=torch.tensor([a['box'] for a in r['annotations']],dtype=torch.float32,device=device)
        labels.append({'class_labels':torch.tensor([a['role'] for a in r['annotations']],dtype=torch.long,device=device),
                       'boxes':torch.cat([(b[:,:2]+b[:,2:])/2,b[:,2:]-b[:,:2]],dim=1)})
    size=cfg['validation_shortest_edge'] if rng is None else int(rng.choice(cfg['training_shortest_edges']))
    inputs=processor(images=images,text=[caption]*len(rows),padding=True,return_tensors='pt',size={'shortest_edge':size,'longest_edge':1333}).to(device)
    maps=build_label_maps(torch.zeros((len(rows),1,256),device=device),inputs.input_ids)
    for i,m in enumerate(maps):
        assert len(m)==len(prompts)
        for role,prompt in enumerate(prompts):
            expected=processor.tokenizer(prompt,add_special_tokens=False)['input_ids']
            positions=torch.where(m[role])[0]
            assert inputs.input_ids[i,positions].tolist()==expected, (rows[0]['scene'],role)
    return inputs,labels


def forward_loss(model, inputs, labels):
    # HF 4.57.6 encoder loss uses learned query embeddings rather than selected
    # encoder proposals, and its proposal boxes are detached. Use the final
    # decoder Hungarian/focal/L1/GIoU criterion only; keep proposal head frozen.
    from transformers.loss.loss_grounding_dino import GroundingDinoForObjectDetectionLoss
    out = model(**inputs)
    options = copy.copy(model.config)
    options.two_stage = False
    options.auxiliary_loss = False
    maps = build_label_maps(out.logits, inputs.input_ids)
    mask = build_text_mask(out.logits, inputs.attention_mask)
    loss, details, _ = GroundingDinoForObjectDetectionLoss(
        out.logits, labels, model.device, out.pred_boxes, options, maps, mask)
    out.loss = loss
    out.loss_dict = details
    return out
