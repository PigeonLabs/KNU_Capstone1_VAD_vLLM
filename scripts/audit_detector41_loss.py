"""Reproduce installed HF encoder-loss caveat on original weights and normal images."""
import json
from pathlib import Path
import torch
from ipad_vad.learned_detector import load_detector,configure_detector,batch_inputs,forward_loss,sha,write


def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    cfg=json.loads(Path('configs/experiment41_detector.json').read_text());rows=[r for r in json.loads(Path(cfg['annotations']).read_text())['records'] if r['status']=='accepted_weak_annotation' and r['partition']=='representation_train' and r['scene']=='R01'][:2]
    m,p,_=load_detector();configure_detector(m);m.cuda();inputs,labels=batch_inputs(rows,p,cfg)
    with torch.no_grad():native=m(**inputs,labels=labels);decoder=forward_loss(m,inputs,labels)
    source=Path('.venv/lib/python3.12/site-packages/transformers/models/grounding_dino/modeling_grounding_dino.py')
    write('results/experiment41/loss_path_audit.json',{'normal_input_ids':[r['id'] for r in rows],'weights':'original_pretrained_not_smoke_or_finetuned','native_loss':float(native.loss),'native_components':{k:float(v) for k,v in native.loss_dict.items()},'final_decoder_loss':float(decoder.loss),'decoder_components':{k:float(v) for k,v in decoder.loss_dict.items()},'implementation_sha256':sha(source),'source_observation':'encoder_pred_boxes are detached reference_points; encoder_logits computed from learned target query embeddings and pre-fusion text_features. Frozen proposal head not optimized by this encoder loss. Final-decoder-only loss uses original matcher and weights, retaining original two-stage inference.','cardinality_error_note':'HF DETR argmax cardinality is not a confidence-thresholded GroundingDINO detection count and is not reported as detector accuracy.'})
    print('Loss-path audit saved')

if __name__=='__main__':main()
