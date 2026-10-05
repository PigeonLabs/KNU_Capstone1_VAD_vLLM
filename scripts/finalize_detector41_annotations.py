"""Apply recorded visual review decisions; never infer approval from a score."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('artifacts/experiment41/annotation_review')
    rows = json.loads((root / 'drafts.json').read_text())
    decisions = json.loads(Path('results/experiment41/annotation_review_decisions.json').read_text())
    manifest = json.loads(Path('results/experiment40/data_manifest.json').read_text())
    data_root = Path(json.loads(Path('configs/experiment40_representation.json').read_text())['data_root'])
    records = []
    seen = Counter()
    for row in rows:
        scene = row['scene']; board = f'{scene}_{seen[scene] // 9:02}.jpg'; seen[scene] += 1
        assert board in decisions['approved_boards'][scene], 'Unreviewed image board'
        assert row['sequence'] in manifest['subsplits'][scene][row['partition']]
        box_list = decisions['overrides'].get(row['id'], row['annotations'])
        for ann in box_list:
            b = np.asarray(ann['box'])
            assert b.shape == (4,) and np.isfinite(b).all() and (b >= 0).all() and (b <= 1).all()
            assert (b[2:] > b[:2]).all() and ann['role'] in range(2 if scene == 'R03' else 3)
        records.append({**row, 'path': str(Path(row['path']).relative_to(data_root)),
                        'image_sha256': sha(row['path']), 'draft_annotations': row['annotations'],
                        'annotations': box_list, 'review_board': board, 'review_board_sha256': sha(root / board),
                        'status': 'excluded' if row['id'] in decisions['excluded'] else 'accepted_weak_annotation',
                        'reviewer': decisions['reviewer'], 'review_note': decisions['excluded'].get(row['id'], decisions['notes'][scene])})
    summary = Counter((r['scene'], r['partition'], r['status']) for r in records)
    report = {'data_root': str(data_root), 'annotation_type': 'model_assisted_weak_boxes_not_human_ground_truth',
              'decisions_sha256': sha('results/experiment41/annotation_review_decisions.json'),
              'source_split_sha256': sha('results/experiment40/data_manifest.json'),
              'summary': [{'scene': s, 'partition': p, 'status': st, 'frames': n} for (s,p,st),n in sorted(summary.items())],
              'records': records}
    Path('results/experiment41/annotations.json').write_text(json.dumps(report, indent=2) + '\n')
    # Show all changed, retained R02 annotations as full images for a final geometry check.
    selected = [r for r in records if r['scene'] == 'R02' and r['id'] in decisions['overrides']]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    for page, start in enumerate(range(0, len(selected), 9)):
        canvas = Image.new('RGB', (1536,1728), 'white'); draw = ImageDraw.Draw(canvas)
        for i,r in enumerate(selected[start:start+9]):
            im = Image.open(data_root/r['path']).convert('RGB').resize((512,512)); d = ImageDraw.Draw(im)
            for a in r['annotations']:
                b = np.array(a['box'])*512; color = ['red','lime','cyan'][a['role']]
                d.rectangle(tuple(b), outline=color,width=2);d.text((b[0],max(0,b[1]-18)),f'r{a["role"]}',font=font,fill=color,stroke_width=1,stroke_fill='black')
            x=i%3*512;y=i//3*576;canvas.paste(im,(x,y));draw.text((x+4,y+514),r['id'],font=font,fill='black')
        canvas.save(root/f'R02_corrected_{page:02}.jpg')
    print(json.dumps(report['summary'],indent=2));print('R02 corrected frames',len(selected))


if __name__ == '__main__':main()
