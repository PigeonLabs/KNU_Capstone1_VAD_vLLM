"""Whole-video normal views and deterministic scene/video balanced batches."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,Sampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


def mobile_transform(augmentation=None):
    if augmentation:
        a=augmentation
        first=[T.RandomResizedCrop(256,scale=a['crop_scale'],ratio=a['crop_ratio'],interpolation=InterpolationMode.BILINEAR),T.ColorJitter(brightness=a['brightness'],contrast=a['contrast'],saturation=a['saturation'],hue=a['hue'])]
    else:first=[T.Resize(256,interpolation=InterpolationMode.BILINEAR),T.CenterCrop(256)]
    return T.Compose([*first,T.ToTensor()])


@lru_cache(maxsize=64)
def decoded(path):
    with Image.open(path) as image:return image.convert('RGB')


def view_image(row):
    image=decoded(row['path']);box=row['box']
    if box is None:return image.copy()
    w,h=image.size
    return image.crop((box[0]*w,box[1]*h,box[2]*w,box[3]*h))


def read_views(path,partitions):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    return [r for r in rows if r['partition'] in partitions]


class ViewDataset(Dataset):
    def __init__(self,rows,transform):self.rows=rows;self.transform=transform
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.transform(view_image(self.rows[index])),index


class AdaptationDataset(Dataset):
    def __init__(self,rows,teacher,augmentation):
        if any(r['partition'] not in ['representation_train','representation_validation'] for r in rows):
            raise ValueError('Calibration or test view in adaptation data')
        self.rows=rows;self.teacher=teacher;self.transform=mobile_transform(augmentation)
        self.scenes={s:i for i,s in enumerate(sorted({r['scene'] for r in rows}))}
        self.videos={v:i for i,v in enumerate(sorted({(r['scene'],r['sequence']) for r in rows}))}
    def __len__(self):return len(self.rows)
    def __getitem__(self,item):
        index,seed=item;row=self.rows[index];image=view_image(row)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);a=self.transform(image);b=self.transform(image)
        return a,b,self.teacher[index],self.scenes[row['scene']],self.videos[(row['scene'],row['sequence'])]


class BalancedBatches(Sampler):
    def __init__(self,rows,partition,samples,batch_size,seed):
        self.rows=rows;self.partition=partition;self.samples=samples;self.batch_size=batch_size;self.seed=seed;self.epoch=0;self.groups={}
        for i,r in enumerate(rows):
            if r['partition']==partition:self.groups.setdefault(r['scene'],{}).setdefault(r['sequence'],{}).setdefault(r['kind'],[]).append(i)
        if len(self.groups)!=4 or batch_size%4 or samples%batch_size:raise ValueError('Four equally represented scenes and full batches required')
        if any(len(v)<2 for v in self.groups.values()):raise ValueError('At least two videos required per scene')
    def __len__(self):return self.samples//self.batch_size
    def __iter__(self):
        rng=np.random.default_rng(self.seed+100003*self.epoch)
        for step in range(len(self)):
            batch=[]
            for scene,videos in sorted(self.groups.items()):
                order=rng.permutation(sorted(videos));count=self.batch_size//4
                for j in range(count):
                    seq=order[j%len(order)];kinds=videos[seq];kind='global' if j%2==0 else 'crop'
                    if kind not in kinds:raise ValueError(f'Missing {kind} views for {scene}/{seq}; do not silently change sampling')
                    index=int(rng.choice(kinds[kind]));seed=int(rng.integers(0,2**31-1));batch.append((index,seed))
            rng.shuffle(batch);yield batch
