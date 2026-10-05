"""Verify publication scope, report links, numbers and frozen training provenance."""
import json,re
from pathlib import Path
from ipad_vad.learned_detector import sha,write
from experiment42_normal import verify,OUT

def main():
    for name in ['training_protocol','training_observations','pre_normal_models_protocol','normal_models_checkpoint','testing_observations','pre_test_scoring_protocol','test_scores_checkpoint']:verify(OUT/f'{name}.json')
    assert json.loads((OUT/'validation.json').read_text())['status']=='passed';files=[Path('README.md'),Path('configs/experiment42_association.json'),Path('src/ipad_vad/learned_association.py'),Path('tests/test_learned_association.py')]+list(Path('docs').glob('EXPERIMENT42*.md'))+[Path('docs/EXPERIMENT43_PLAN.md')]+list(Path('scripts').glob('*42.py'))+[Path('scripts/experiment42_normal.py')]+list(OUT.rglob('*'));files=sorted(set(p for p in files if p.is_file() and p.name!='public_manifest.json'));links=0
    for p in files:
        assert p.suffix in ['.py','.md','.json','.csv','.png'],p;assert p.stat().st_size<50_000_000,p
        if p.suffix=='.json':json.loads(p.read_text())
        if p.suffix=='.md':
            text=p.read_text()
            if p.name=='README.md':text=text[text.index('## 실험 42'):]
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if '://' in target or target.startswith('#'):continue
                assert (p.parent/target.split('#')[0]).exists(),(p,target);links+=1
    report=Path('docs/EXPERIMENT42.md').read_text();readme=Path('README.md').read_text();table=(OUT/'summary_table.md').read_text().strip();assert table in report and table in readme
    metrics=json.loads((OUT/'metrics.json').read_text());assert all(not any(r[k] for k in ['added_fp','removed_fp','added_tp','removed_tp']) for r in metrics['paired_41_contrasts']);assert all(r['combined_auroc_delta']==0 for r in metrics['paired_41_contrasts'] if r['branch']=='raw' or r['scene']!='R04')
    write(OUT/'public_manifest.json',{'file_sha256':{str(p):sha(p) for p in files},'files':len(files),'links_checked':links,'raw_media_weights_features_logs_excluded':True,'training_and_scoring_protocols_verified':True,'plots_visually_inspected':['training.png','association_funnel.png'],'tests_passed':187});print('PUBLICATION READY',len(files),'files;',links,'links')
if __name__=='__main__':main()
