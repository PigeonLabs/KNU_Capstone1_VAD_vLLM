"""Render the 24 preselected normal boundaries for experiment34 locally."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from experiment34_semantic_priority import OUT,ART,root,load,sha,write,configure_common,common

def main():
    configure_common();common.verify_freeze('pre_normal_protocol.json');common.verify_freeze('pre_calibration_checkpoint.json')
    selection=Path('results/experiment33/case_selection.json');contexts=Path('results/experiment33/case_context.json');checkpoint=json.loads(Path('results/experiment33/pre_visual_checkpoint.json').read_text())
    assert sha(selection)==checkpoint['selection_sha256'] and sha(contexts)==checkpoint['case_context_sha256']
    cases=json.loads(contexts.read_text())['cases'];cache={};records=[];sources={}
    for case in cases:
        c=case['case'];seq=c['sequence']
        if seq not in cache:cache[seq]={g:load(root(g)/f'training_{seq}.npz') for g in ['control','semantic']}
        frames=[]
        for frame in case['frames']:
            path=frame['source_path'];assert sha(path)==checkpoint['source_normal_image_sha256'][path];sources[path]=sha(path);i=frame['sample'];d=cache[seq]['control'];new=cache[seq]['semantic'];oldids=d['relation_detection_indices'][i];newids=new['relation_detection_indices'][i]
            frames.append({'source_path':path,'sample':i,'source_frame':int(d['indices'][i]),'old_phase':int(d['phases'][i]),'new_phase':int(new['phases'][i]),'valid':bool(d['relation_valid'][i]),'old_selected':oldids.tolist(),'new_selected':newids.tolist(),'old_tracks':[int(d['tracks'][x]) if x>=0 else None for x in oldids],'new_tracks':[int(d['tracks'][x]) if x>=0 else None for x in newids],'anchor_changed':bool(oldids[0]!=newids[0])})
        records.append({'case':c,'frames':frames})
    write(OUT/'visual_case_context.json',{'normal_only':True,'case_selection_source':str(selection),'selection_sha256':sha(selection),'cases':records})
    write(OUT/'pre_visual_checkpoint.json',{'case_context_sha256':sha(OUT/'visual_case_context.json'),'source_normal_image_sha256':sources,'renderer_sha256':sha(__file__),'local_feature_sha256':{str(root(g)/f'training_{seq}.npz'):sha(root(g)/f'training_{seq}.npz') for seq in cache for g in ['control','semantic']}})
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);dest=ART/'contact_sheets';dest.mkdir(parents=True,exist_ok=True);pages=[]
    for start in range(0,len(records),2):
        sheet=Image.new('RGB',(1500,1080),'#f5f7f9');draw=ImageDraw.Draw(sheet);draw.text((12,8),'Anchor: OLD red / NEW blue / SAME purple | Target cyan | Next frame is posthoc context',font=font,fill='black')
        for row,record in enumerate(records[start:start+2]):
            c=record['case'];d=cache[c['sequence']]['control'];top=43+row*514;draw.text((12,top),f'{c["case_id"]} {c["partition"]} | '+','.join(c['groups']),font=font,fill='black')
            for col,f in enumerate(record['frames']):
                x=col*500+12;y=top+27;sheet.paste(Image.open(f['source_path']).convert('RGB').resize((384,384)),(x,y))
                ids=[(f['old_selected'][0],'#c52f43','OLD'),(f['new_selected'][0],'#2360e0','NEW')] if f['anchor_changed'] else [(f['old_selected'][0],'#aa38be','SAME')]
                ids.append((f['old_selected'][1],'#00a9bb','T'))
                for idx,color,label in ids:
                    if idx<0:continue
                    a,b,c0,d0=d['boxes'][idx];rect=(x+a*384,y+b*384,x+c0*384,y+d0*384);draw.rectangle(rect,outline=color,width=3);draw.text((max(x,rect[0]),max(y,rect[1]-16)),f'{label} t{d["tracks"][idx]}',font=small,fill=color)
                yy=y+388;draw.text((x,yy),f'frame {f["source_frame"]} phase {f["old_phase"]}->{f["new_phase"]} valid={f["valid"]}',font=small,fill='black')
                for j,(name,index) in enumerate([('OLD',f['old_selected'][0]),('NEW',f['new_selected'][0])]):
                    line=f'{name}: missing' if index<0 else f'{name}: t{d["tracks"][index]} margin={d["anchor_gate_margin"][index]:.5f}'
                    draw.text((x,yy+18*(j+1)),line,font=small,fill='black')
        path=dest/f'cases_{start//2+1:02}.jpg';sheet.save(path,quality=95);pages.append({'path':str(path),'sha256':sha(path),'case_ids':[r['case']['case_id'] for r in records[start:start+2]]})
    write(OUT/'local_contact_sheet_manifest.json',{'local_only':True,'pages':pages});print('Rendered',len(pages),'pages, cases',len(records),flush=True)

if __name__=='__main__':main()
