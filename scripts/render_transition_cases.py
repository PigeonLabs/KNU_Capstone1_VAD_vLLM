"""Local scientific evidence contact sheets; source images are never published."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from audit_missing_age import sha


def main():
    out=Path('results/experiment29');art=Path('artifacts/experiment29');dest=art/'contact_sheets';dest.mkdir(parents=True,exist_ok=True);selection=json.loads((out/'case_selection.json').read_text());check=json.loads((out/'pre_visual_checkpoint.json').read_text());assert sha(out/'case_selection.json')==check['selection_sha256']
    for p,h in check['source_normal_image_sha256'].items():assert sha(p)==h,p
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);pages=[]
    for group in ['target_3_to_2','control_previous_3']:
        cases=[c for c in selection['cases'] if c['selection_group']==group]
        for start in range(0,len(cases),2):
            subset=cases[start:start+2];sheet=Image.new('RGB',(1440,850),'#f4f6f8');draw=ImageDraw.Draw(sheet);draw.text((12,7),f'{group} | Red: selected anchor; cyan: selected target | next sample: posthoc only',font=font,fill='#17212b')
            for row,case in enumerate(subset):
                trace=json.loads((art/'traces'/f'{case["sequence"]}.json').read_text());i=case['sample'];y=40+row*402;draw.text((12,y),f'{case["case_id"]}  latent {case["previous_phase"]}->{case["current_phase"]}',font=font,fill='black')
                for col,j in enumerate([i-1,i,i+1]):
                    if j>=len(trace['frames']):continue
                    frame=trace['frames'][j];im=Image.open(frame['source_path']).convert('RGB');im.thumbnail((468,315));x=col*480+6;top=y+27;sheet.paste(im,(x,top));draw.text((x,top+318),f'frame {frame["source_frame"]} phase {frame["phase"]} valid={frame["relation_valid"]}',font=small,fill='black')
                    for role,color in [('anchor','#ef3340'),('target','#00a5bb')]:
                        box=frame['selected'][role]
                        if box is None:continue
                        a,b,c,d=box['bbox_normalized'];rect=(x+a*im.width,top+b*im.height,x+c*im.width,top+d*im.height);draw.rectangle(rect,outline=color,width=3);draw.text((max(x,rect[0]),max(top,rect[1]-18)),f'{role[0].upper()} tr{box["track"]}',font=small,fill=color)
                    anchor=frame['selected']['anchor'];target=frame['selected']['target'];draw.text((x,top+340),f'A track={None if anchor is None else anchor["track"]}; T track={None if target is None else target["track"]}',font=small,fill='black');draw.text((x,top+359),f'A raw/gate margin: {"missing" if anchor is None else format(anchor["raw_semantic_margin"],".4f")+" / "+format(anchor["temporal_semantic_margin"],".4f")}',font=small,fill='black')
            path=dest/f'{group}_{start//2+1:02}.jpg';sheet.save(path,quality=94);pages.append({'path':str(path),'case_ids':[c['case_id'] for c in subset],'sha256':sha(path)})
    (out/'local_contact_sheet_manifest.json').write_text(json.dumps({'local_only':True,'renderer_sha256':sha('scripts/render_transition_cases.py'),'pages':pages},indent=2)+'\n');print(json.dumps(pages,indent=2))


if __name__=='__main__':main()
