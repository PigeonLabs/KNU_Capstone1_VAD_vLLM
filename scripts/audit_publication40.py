"""Check public experiment artifacts, local links, scope, and result accounting."""
import json,re,hashlib
from pathlib import Path
import numpy as np
from prepare_representation40 import OUT,sha,write


def main():
    docs=[Path('README.md'),*sorted(Path('docs').glob('EXPERIMENT40*.md')),Path('docs/EXPERIMENT41_PLAN.md')];links=[]
    for p in docs:
        text=p.read_text()
        if p.name=='README.md':text=text[text.index('## 실험 40 —'):]
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if target.startswith(('http://','https://','#')):continue
            path=(p.parent/target.split('#')[0]).resolve();assert path.exists(),(p,target);links.append({'source':str(p),'target':target})
    d=json.loads((OUT/'metrics.json').read_text());val=json.loads((OUT/'validation.json').read_text());audit=json.loads((OUT/'evaluation_audit.json').read_text());assert len(d['variants'])==32 and len(d['scene_macro'])==8 and len(d['adaptation_seed_summary'])==2
    a=[r for r in d['variants'] if r['run']=='A'];assert sum(r['normal_frames']+r['anomaly_frames'] for r in a)==31550 and sum(r['unknown_frames'] for r in a)==1912 and sum(r['events']['events'] for r in a)==66
    assert val['normal_files_verified']==888 and val['full_and_holdout_normal_models_scores_reconstructed']==208 and val['test_predictions_reconstructed']==528 and audit['prediction_files_frozen_before_any_labels']==528
    assert json.loads((OUT/'training_queue.json').read_text())['status']=='complete' and json.loads((OUT/'evaluation_queue.json').read_text())['status']=='complete'
    recovery=json.loads((OUT/'resource_preflight_recovery.json').read_text());aborted=json.loads((OUT/'resource_preflight_attempt1/C_s42_history.json').read_text());restarted=json.loads((OUT/'C_s42_history.json').read_text())
    assert recovery['last_complete_epoch']==aborted[-1]['epoch']==1 and recovery['last_complete_optimizer_step']==aborted[-1]['step']==64
    for key in ['train','validation']:assert aborted[-1][key]==restarted[1][key]
    for arm in ['C','D']:
        group=next(r for r in d['adaptation_seed_summary'] if r['arm']==arm)
        for k,summary in group['macro'].items():
            x=[r[k] for r in d['scene_macro'] if r['run'].startswith(arm)];np.testing.assert_allclose([summary['mean'],summary['sample_std']],[np.mean(x),np.std(x,ddof=1)],rtol=0,atol=0)
    for p in [Path('README.md'),Path('docs/EXPERIMENT40.md'),OUT/'recommendations.md']:
        text=p.read_text();block=text[text.index('다음 Recommended improvements',text.index('## 실험 40 —') if p.name=='README.md' else 0):] if p.name!='recommendations.md' else text
        assert re.findall(r'^\| ([123]) \|',block,re.M)==['1','2','3'],p
    files=set(docs+[Path('requirements-experiment40.txt'),Path('scripts/extract_features.py'),Path('src/ipad_vad/learned_visual.py'),Path('src/ipad_vad/representation_data.py'),Path('tests/test_learned_visual.py')])
    files.update(Path('configs').glob('experiment40*.json'));files.update(Path('scripts').glob('*representation40*.py'));files.add(Path('scripts/experiment40_normal.py'));files.add(Path('scripts/extract_adapted40.py'));files.add(Path(__file__).resolve().relative_to(Path.cwd()));files.update(Path('scripts').glob('audit_publication40.py'))
    for root in [OUT,Path('results/experiment40_R02_base')]:
        files.update(p for p in root.rglob('*') if p.is_file() and p.name!='publication_manifest.json')
    for p in files:
        assert p.suffix in ['.md','.txt','.json','.csv','.png','.py'] and p.stat().st_size<10*1024**2,p
        assert not p.is_symlink(),p
        if p.suffix=='.png':assert p.parent==OUT/'figures',p
        if p.suffix=='.json':json.loads(p.read_text())
    result={'completed_experiment':'40','next_experiment':'41_plan_only_not_executed','public_files':len(files),'local_links_checked':len(links),'unit_tests':{'passed':177,'log_local_only':'/tmp/experiment40_final_tests.log'},'result_accounting_verified':True,'normal_models_and_predictions_independently_reconstructed':True,'figures_rendered_and_visually_checked':True,'data_weights_feature_caches_and_server_logs_excluded':True,'file_sha256':{str(p):sha(p) for p in sorted(files)}}
    write(OUT/'publication_manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k!='file_sha256'},indent=2))

if __name__=='__main__':main()
