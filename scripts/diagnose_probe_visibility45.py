"""Check whether native preprocessing removes applied synthetic corruptions."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoImageProcessor
from ipad_vad.large_visual import probe_image
from ipad_vad.representation_data import view_image
from candidates45_common import OUT,ART,config,guard,sha,write


def main():
    guard();torch.set_num_threads(2)
    rows=json.loads((ART/'probe_views.json').read_text())['validation'];variants=['occlusion','local_shuffle','local_noise'];results=[]
    for name in config()['candidates']:
        record=json.loads((OUT/f'{name}_model.json').read_text())
        processor=AutoImageProcessor.from_pretrained(record['local_path'],local_files_only=True,use_fast=False)
        buckets={}
        for row in rows:
            image=view_image(row);images=[image]+[probe_image(image,v,row['probe_seed']) for v in variants]
            pixels=processor(images=images,return_tensors='pt')['pixel_values'].numpy()
            for j,variant in enumerate(variants,1):
                raw=float(np.any(np.array(images[j])!=np.array(image),axis=-1).mean())
                processed=float(np.any(np.abs(pixels[j]-pixels[0])>1e-6,axis=0).mean())
                buckets.setdefault((row['scene'],row['kind'],variant),[]).append((raw,processed))
        for (scene,kind,variant),values in buckets.items():
            a=np.asarray(values);results.append({'candidate':name,'scene':scene,'kind':kind,'corruption':variant,
                'views':len(a),'raw_no_change':int((a[:,0]==0).sum()),'processed_no_change':int((a[:,1]==0).sum()),
                'changed_raw_but_removed_by_processor':int(((a[:,0]>0)&(a[:,1]==0)).sum()),
                'mean_raw_changed_pixel_fraction':float(a[:,0].mean()),'mean_processed_changed_pixel_fraction':float(a[:,1].mean())})
        print('Visibility checked',name,flush=True)
    write(OUT/'probe_visibility.json',{'normal_only':True,'selection_changed':False,'rows':results,
        'probe_protocol_sha256':sha(OUT/'probe_protocol.json'),'selection_sha256':sha(OUT/'selection.json'),
        'script_sha256':sha(__file__),'note':'Applied synthetic pixels surviving each native processor; not perceptual difficulty or actual anomaly labels. Duplicate sampled views are not independent observations.'})


if __name__=='__main__':main()
