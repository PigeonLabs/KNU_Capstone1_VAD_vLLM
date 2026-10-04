"""Use only normal FIT frames with a loopback Qwen vision endpoint."""
import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
from ipad_vad.data import frames_in_order

PROMPT = '''You are documenting only the VISIBLE normal operation of an industrial device.
The supplied images are two normal training sequences, each sampled in temporal order.
Use no test data or imagined defects. Do not infer hidden actions such as grasping if not visible.
Return ONLY a JSON object, no markdown. Schema:
{"objects":[{"id":"short_ascii_role","detection_prompt":"short concrete English noun phrase","description":"visible role"}],
"phases":[{"id":"short_ascii_state","description":"A photo of ... (a concrete visually discriminable normal state)"}],
"normal_order":["phase_id"],"cyclic":true,"uncertainties":["limitations"]}.
Use 1-3 object roles (include transported product and relevant machine part), 2-4 observable phases.
Phases must describe distinct VISIBLE positions/states, not defect types. Vocabulary must work as object detection prompts.
When a fine-grained action is not observable, use spatial progression states instead. Do not invent numeric period or FPS.'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/experiment01.json'))
    args=p.parse_args();cfg=json.loads(args.config.read_text());scene=cfg['scene']
    url=cfg['local_vlm_url']
    if not url.startswith('http://127.0.0.1:'): raise ValueError('Local loopback endpoint required')
    splits=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    content=[{'type':'text','text':PROMPT}];sources=[]
    for seq in splits['fit'][:2]:
        frames=frames_in_order(args.data_root/scene/'training/frames'/seq)
        ids=np.linspace(0,len(frames)-1,6,dtype=int)
        for i in ids:
            name=f'{scene}/training/frames/{seq}/{frames[i].name}';sources.append(name)
            content.append({'type':'text','text':f'Normal sequence {seq}, source frame {i} (ordered within this sequence):'})
            content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(frames[i].read_bytes()).decode()}})
    payload={'messages':[{'role':'user','content':content}]}
    request=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start=time.perf_counter()
    with urllib.request.urlopen(request,timeout=1800) as response: body=json.load(response)
    elapsed=time.perf_counter()-start
    text=body['choices'][0]['message']['content'];begin=text.find('{');end=text.rfind('}')
    process=json.loads(text[begin:end+1])
    if not 1<=len(process['objects'])<=3 or not 2<=len(process['phases'])<=4:
        raise ValueError('Unexpected vocabulary/phase cardinality')
    phase_ids=[p['id'] for p in process['phases']]
    if set(process['normal_order'])!=set(phase_ids): raise ValueError('Phase order mismatch')
    out=Path('results/experiment01');out.mkdir(parents=True,exist_ok=True)
    result={'scene':scene,'sources':sources,'prompt':PROMPT,'process':process,
            'elapsed_seconds':elapsed,'usage':body.get('usage'),'finish_reason':body['choices'][0].get('finish_reason'),
            'note':'Generated normal-state hypotheses, not ground truth. Reasoning text is not persisted.'}
    if result['finish_reason'] not in ('stop', None): raise ValueError('Generation did not finish normally')
    (out/'process_discovery.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
