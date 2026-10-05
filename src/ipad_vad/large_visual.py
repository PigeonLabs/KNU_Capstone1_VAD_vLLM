"""Official large vision towers with their checkpoint-native preprocessing."""
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageEnhance


class LargeVisual(nn.Module):
    def __init__(self, model, family):
        super().__init__();self.model=model;self.family=family

    def forward(self, pixels):
        out=self.model(pixel_values=pixels)
        if self.family=='clip':return out.image_embeds
        if self.family=='siglip':return out.pooler_output
        if self.family=='dinov2':return out.last_hidden_state[:,0]
        raise ValueError(self.family)


def load_large_visual(record):
    from transformers import AutoImageProcessor, CLIPVisionModelWithProjection, SiglipVisionModel, Dinov2Model
    family=record['family'];cls={'clip':CLIPVisionModelWithProjection,'siglip':SiglipVisionModel,'dinov2':Dinov2Model}[family]
    model,info=cls.from_pretrained(record['local_path'],local_files_only=True,
        output_loading_info=True,attn_implementation='sdpa')
    if info['missing_keys'] or info['mismatched_keys'] or info['error_msgs']:
        raise RuntimeError('Uninitialized or incompatible visual weights: '+str(info))
    # Full CLIP/SigLIP archives contain unused text parameters. Visual keys must all load.
    if any('vision' in k or 'visual_projection' in k for k in info['unexpected_keys']):
        raise RuntimeError('Unexpected visual checkpoint keys: '+str(info['unexpected_keys']))
    processor=AutoImageProcessor.from_pretrained(record['local_path'],local_files_only=True,use_fast=False)
    return LargeVisual(model,family),processor,info


def probe_image(image, variant, seed, area=.16):
    """Model-independent normal-image proxy; never represents ground-truth defects."""
    if variant=='clean':return image.copy()
    if variant=='benign':return ImageEnhance.Brightness(image).enhance(1.05)
    rng=np.random.default_rng(seed);a=np.array(image.convert('RGB')).copy();h,w=a.shape[:2]
    ph=max(1,min(h,round(h*np.sqrt(area))));pw=max(1,min(w,round(w*np.sqrt(area))))
    y=int(rng.integers(0,h-ph+1));x=int(rng.integers(0,w-pw+1));patch=a[y:y+ph,x:x+pw].copy()
    if variant=='occlusion':patch[:]=np.round(a.mean(axis=(0,1))).astype(np.uint8)
    elif variant=='local_shuffle':patch=np.roll(patch,(max(1,ph//2),max(1,pw//2)),axis=(0,1))
    elif variant=='local_noise':patch=np.clip(.5*patch+.5*rng.integers(0,256,patch.shape),0,255).astype(np.uint8)
    else:raise ValueError(variant)
    a[y:y+ph,x:x+pw]=patch
    return Image.fromarray(a)
