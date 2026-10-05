"""Validate finished phase-learning publication and exact upload scope."""
import json,re
from pathlib import Path
from ipad_vad.learned_detector import sha,write
from experiment43_normal import OUT,verify

def main():
    for name in ['training_protocol','training_observations','pre_normal_models_protocol','normal_models_checkpoint','testing_observations','pre_test_scoring_protocol','test_scores_checkpoint']:verify(OUT/f'{name}.json')
    assert json.loads((OUT/'validation.json').read_text())['status']=='passed';training=json.loads((OUT/'training.json').read_text());assert len(training['rows'])==8 and all(r['optimizer_steps']==960 for r in training['rows']);metrics=json.loads((OUT/'metrics.json').read_text());assert len(metrics['variants'])==96;assert len(metrics['paired_teacher_contrasts'])==64
    files=[Path('README.md'),Path('src/ipad_vad/learned_phase.py'),Path('tests/test_learned_phase.py'),Path('docs/LEARNED_PIPELINE_SUMMARY.md'),Path('docs/EXPERIMENT44_PLAN.md')]+list(Path('docs').glob('EXPERIMENT43*.md'))+list(Path('scripts').glob('*43*.py'))+list(Path('configs').glob('experiment43_*.json'))+list(OUT.rglob('*'));files=sorted(set(p for p in files if p.is_file() and p.name!='public_manifest.json'));links=0
    for p in files:
        assert p.suffix in ['.py','.md','.json','.csv','.png'],p;assert p.stat().st_size<50_000_000,p
        if p.suffix=='.json':json.loads(p.read_text())
        if p.suffix=='.md':
            text=p.read_text()
            if p.name=='README.md':text=text[text.index('## 실험 43'):]
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if '://' in target or target.startswith('#'):continue
                assert (p.parent/target.split('#')[0]).exists(),(p,target);links+=1
    table=(OUT/'summary_table.md').read_text().strip();assert table in Path('README.md').read_text() and table in Path('docs/EXPERIMENT43.md').read_text()
    write(OUT/'public_manifest.json',{'file_sha256':{str(p):sha(p) for p in files},'files':len(files),'links_checked':links,'raw_media_weights_features_logs_excluded':True,'training_scoring_protocols_verified':True,'plots_visually_inspected':['training.png','process_comparison.png'],'tests_passed':190,'next44_is_plan_without_independent_data_or_results':True});print('PUBLICATION READY',len(files),'files;',links,'links')
if __name__=='__main__':main()
