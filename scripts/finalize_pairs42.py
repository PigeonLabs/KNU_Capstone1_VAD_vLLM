"""Freeze supported weak supervision after visual candidate inspection."""
import csv,json
from pathlib import Path
from collections import Counter
from ipad_vad.learned_detector import sha,write
OUT=Path('results/experiment42')
def main():
    ds=json.loads((OUT/'positive_pair_review_drafts.json').read_text())
    for i,r in enumerate(ds):
        r['reviewer']='Codex visual inspection; not human identity ground truth'
        r['observation']='Same visible object/fixture or assembled cargo across adjacent samples; weak temporal consistency, not independent identity GT.'
        if 32<=i<=35:r['observation']='Material box may aggregate cut pieces or change physical coverage; exclude ALL R04 material automatic positives.'
        if i>=40:r['observation']='Broad stationary apparatus crop includes changing contents; used only as weak region/fixture consistency.'
        r['used_role']=not (r['scene']=='R04' and r['role']=='0')
    write(OUT/'positive_pair_review.json',{'draft_sha256':sha(OUT/'positive_pair_review_drafts.json'),'reviewed_boards':{str(p):sha(p) for p in sorted(Path('artifacts/experiment42/pair_review').glob('positive_*.jpg'))},'sample_policy':'Worst confidence and lowest IoU per scene-role/train-or-val; 44 pairs. Stress audit, not random error-rate estimate; other pairs NOT individually inspected.','excluded_positive_roles':[['R04',0]],'pairs':ds})
    accepted={r['id'] for r in json.loads((OUT/'negative_pair_review.json').read_text())['pairs'] if r['status']=='accept'};rows=list(csv.DictReader((OUT/'pair_candidates.csv').open()));keep=[]
    for r in rows:
        if r['kind']=='negative_candidate' and r['id'] in accepted:keep.append(r)
        elif r['kind']=='positive_candidate' and not(r['scene']=='R04' and r['role']=='0'):keep.append(r)
    with (OUT/'accepted_pairs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(keep)
    counts=Counter(r['partition']+'/'+r['kind'] for r in keep)
    write(OUT/'pair_selection.json',{'counts':dict(counts),'all_R04_material_automatic_positives_excluded':True,'calibration_excluded_from_gradient_and_checkpoint':True,'accepted_sha256':sha(OUT/'accepted_pairs.csv'),'review_sha256':{str(OUT/n):sha(OUT/n) for n in ['negative_pair_review.json','positive_pair_review.json']},'limitations':'Shared temporal-consistency supervision covers all processes; direct same-role identity separation only R04 material. Negative validation: 5 frames across 4 videos; not identity GT. No R04 material temporal-positive recall validation.'})
    print(dict(counts))
if __name__=='__main__':main()
