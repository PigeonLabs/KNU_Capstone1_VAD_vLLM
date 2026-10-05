"""Shared visual adaptation; no phase, track, or anomaly-label supervision."""
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0:
            raise ValueError('A linear base and positive rank are required')
        self.base = base
        self.scale = alpha / rank
        self.base.requires_grad_(False)
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


def configure_adaptation(visual, mode, rank=8, alpha=16):
    if mode not in ['frozen', 'lora', 'full']:
        raise ValueError('Unknown visual adaptation mode')
    visual.requires_grad_(mode == 'full')
    targets = []
    if mode == 'lora':
        for name, module in list(visual.named_modules()):
            if isinstance(module, nn.Linear) and (name.endswith('token_mixer.qkv') or name.endswith('token_mixer.proj') or name == 'trunk.head.fc'):
                parent_name, leaf = name.rsplit('.', 1)
                setattr(visual.get_submodule(parent_name), leaf, LoRALinear(module, rank, alpha))
                targets.append(name)
        if not targets:
            raise ValueError('No declared MobileCLIP2 LoRA target matched')
    # Full FT means all visual parameters, with common fixed BN running statistics.
    visual.eval()
    return {'mode': mode, 'lora_targets': targets,
            'trainable_parameters': sum(p.numel() for p in visual.parameters() if p.requires_grad),
            'total_parameters': sum(p.numel() for p in visual.parameters()),
            'trainable_names': [n for n, p in visual.named_parameters() if p.requires_grad],
            'batchnorm_running_statistics': 'frozen'}


def load_mobile_visual(model_record):
    # Pinned OpenCLIP 3.3.0 internal factory; never loads the unused text tower.
    from open_clip.model import _build_vision_tower
    from safetensors import safe_open
    path = Path(model_record['local_path'])
    cfg = json.loads((path / 'open_clip_config.json').read_text())
    visual = _build_vision_tower(cfg['model_cfg']['embed_dim'], cfg['model_cfg']['vision_cfg'])
    with safe_open(path / 'open_clip_model.safetensors', framework='pt', device='cpu') as f:
        state = {k.removeprefix('visual.'): f.get_tensor(k) for k in f.keys() if k.startswith('visual.')}
    visual.load_state_dict(state, strict=True)
    return visual, cfg['preprocess_cfg']


def representation_loss(z1, z2, teacher, scenes, videos, temperature=.1, contrastive_weight=.1):
    z1, z2, teacher = [F.normalize(z.float(), dim=-1) for z in [z1, z2, teacher]]
    n = len(z1)
    positive = torch.eye(n, dtype=torch.bool, device=z1.device)
    negatives = (scenes[:, None] == scenes[None, :]) & (videos[:, None] != videos[None, :])
    if not bool(negatives.any(1).all()):
        raise ValueError('Each item requires a different-video negative from the same scene')
    logits = (z1 @ z2.T / temperature).masked_fill(~(positive | negatives), -torch.inf)
    labels = torch.arange(n, device=z1.device)
    contrast = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    preservation = 1 - .5 * ((z1 * teacher.detach()).sum(-1).mean() + (z2 * teacher.detach()).sum(-1).mean())
    return preservation + contrastive_weight * contrast, {'teacher_cosine_loss': preservation.detach(), 'contrastive_loss': contrast.detach()}
