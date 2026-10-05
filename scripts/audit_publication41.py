"""Check public experiment41 files, result accounting and report links."""
import json,re
from pathlib import Path
import numpy as np
from ipad_vad.learned_detector import sha,write

OUT=Path('results/experiment41')

def main():
    docs=[Path('README.md'),*sorted(Path('docs').glob('EXPERIMENT41*.md')),Path('docs/EXPERIMENT42_PLAN.md')];links=[]
    for p in docs:
        text=p.read_text()
        if p.name=='README.md':text=text[text.index('## 실험 41 —'):]
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if target.startswith(('http://','https://','#')):continue
            assert (p.parent/target.split('#')[0]).resolve().exists(),(p,target);links.append((str(p),target))
    metrics=json.loads((OUT/'metrics.json').read_text());val=json.loads((OUT/'validation.json').read_text());audit=json.loads((OUT/'evaluation_audit.json').read_text());assert len(metrics['variants'])==32 and len(metrics['scene_macro'])==8 and len(metrics['paired_frozen_detector_contrasts'])==32
    rows=[r for r in metrics['variants'] if r['run']=='A'];assert sum(r['normal_frames']+r['anomaly_frames'] for r in rows)==31550 and sum(r['unknown_frames'] for r in rows)==1912 and sum(r['events']['events'] for r in rows)==66
    assert val['normal_files_verified']==888 and val['full_and_holdout_normal_models_scores_reconstructed']==208 and val['test_predictions_reconstructed']==528 and audit['prediction_files_frozen_before_any_labels']==528
    for arm in ['C','D']:
        summary=next(r for r in metrics['adaptation_seed_summary'] if r['arm']==arm)
        for k,v in summary['macro'].items():
            a=[r[k] for r in metrics['scene_macro'] if r['run'].startswith(arm)];np.testing.assert_allclose([v['mean'],v['sample_std']],[np.mean(a),np.std(a,ddof=1)],atol=0,rtol=0)
    for p in [Path('README.md'),Path('docs/EXPERIMENT41.md'),OUT/'recommendations.md']:
        text=p.read_text();offset=text.index('## 실험 41 —') if p.name=='README.md' else 0
        block=text[text.index('다음 Recommended improvements',offset):] if p.name!='recommendations.md' else text
        assert re.findall(r'^\| ([123]) \|',block,re.M)==['1','2','3'],p
    tests=Path('artifacts/experiment41/final_tests.log').read_text();assert re.search(r'182 passed',tests)
    files=set(docs+[Path('src/ipad_vad/learned_detector.py'),Path('src/ipad_vad/optional_dwell.py'),Path('tests/test_learned_detector.py'),Path('tests/test_optional_dwell.py')]);files.update(Path('configs').glob('experiment41*.json'));files.update(Path('scripts').glob('*41*.py'));files.update(p for p in OUT.rglob('*') if p.is_file() and p.name!='publication_manifest.json')
    for p in files:
        assert p.suffix in ['.md','.json','.csv','.png','.py'] and p.stat().st_size<10*1024**2 and not p.is_symlink(),p
        if p.suffix=='.png':assert p.parent==OUT/'figures',p
        if p.suffix=='.json':json.loads(p.read_text())
    write(OUT/'publication_manifest.json',{'completed_experiment':'41','next_experiment':'42_plan_only','public_files':len(files),'local_links_checked':len(links),'tests_passed':182,'figures_visually_reviewed':True,'normal_holdout_boards_visually_reviewed':8,'result_accounting_verified':True,'normal_models_and_predictions_reconstructed':True,'data_weights_feature_caches_and_server_logs_excluded':True,'file_sha256':{str(p):sha(p) for p in sorted(files)}})
    print('Publication audit:',len(files),'files;',len(links),'links')

if __name__=='__main__':main()
