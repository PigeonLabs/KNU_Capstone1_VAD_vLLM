"""Local contact sheets for frozen normal cases; do not publish source pixels."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from experiment33_observation_trace import OUT,ART,guard,sha,write,verify_protocol


def main():
    allowed,opened=guard();verify_protocol();check=json.loads((OUT/'pre_visual_checkpoint.json').read_text());assert sha(OUT/'case_selection.json')==check['selection_sha256'];assert sha(OUT/'case_context.json')==check['case_context_sha256']
    for p,h in check['source_normal_image_sha256'].items():allowed.add(str(Path(p).resolve()));assert sha(p)==h
    cases=json.loads((OUT/'case_context.json').read_text())['cases'];dest=ART/'contact_sheets';dest.mkdir(parents=True,exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);pages=[]
    for start in range(0,len(cases),2):
        subset=cases[start:start+2];sheet=Image.new('RGB',(1500,1140),'#f5f7f9');draw=ImageDraw.Draw(sheet);draw.text((12,8),'Selected: RED anchor / CYAN target; other cached candidates: gray | next frame is posthoc context only',font=font,fill='black')
        for row,case in enumerate(subset):
            top=43+row*544;c=case['case'];draw.text((12,top),f'{c["case_id"]} {c["partition"]} | '+','.join(c['groups']),font=font,fill='black')
            for col,f in enumerate(case['frames']):
                x=col*500+12;y=top+27;im=Image.open(f['source_path']).convert('RGB').resize((384,384));sheet.paste(im,(x,y))
                for role,color in [('anchor','#ef3340'),('target','#00a9bb')]:
                    for candidate in f['roles'][role]['candidates']:
                        a,b,c0,d=candidate['bbox_normalized'];rect=(x+a*384,y+b*384,x+c0*384,y+d*384);cl=color if candidate['selected'] else '#b0b0b0';draw.rectangle(rect,outline=cl,width=3 if candidate['selected'] else 1);draw.text((max(x,rect[0]),max(y,rect[1]-16)),f'{role[0].upper()}{candidate["detection_index"]}/t{candidate["track"]}',font=small,fill=cl)
                yy=y+388;draw.text((x,yy),f'frame {f["source_frame"]} phase {f["phase"]} valid={f["relation_valid"]}',font=small,fill='black')
                for j,role in enumerate(['anchor','target']):
                    r=f['roles'][role];draw.text((x,yy+18*(j+1)),f'{role[0].upper()}: {r["reason"]} raw={r["raw_count"]} joint={r["joint_pass_count"]}',font=small,fill='black')
                for j,a in enumerate(f['roles']['anchor']['candidates']):
                    draw.text((x,yy+54+17*j),f'A{a["detection_index"]}/t{a["track"]} area={a["area"]:.3f} raw/gate={a["raw_margin"]:.4f}/{a["gate_margin"]:.4f}',font=small,fill='black')
        p=dest/f'cases_{start//2+1:02}.jpg';sheet.save(p,quality=95);pages.append({'path':str(p),'case_ids':[x['case']['case_id'] for x in subset],'sha256':sha(p)})
    write(OUT/'local_contact_sheet_manifest.json',{'local_only':True,'renderer_sha256':sha('scripts/render_observation_cases.py'),'pages':pages});write(OUT/'render_access_log.json',{'opened_data_paths':sorted(opened),'test_data_opened':False});print(json.dumps(pages,indent=2))


if __name__=='__main__':main()
