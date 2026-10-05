"""Check scientific evidence, generated tables, local links and publication scope."""
import json,re
from pathlib import Path
from candidates45_common import OUT,config,verify_record,sha,write


def main():
    for name in ['probe_protocol','selection_checkpoint','pre_normal_models_protocol','normal_models_checkpoint','test_execution_protocol',
                 'pre_test_scoring_protocol','test_scores_checkpoint']:
        verify_record(OUT/f'{name}.json')
    for name in config()['candidates']:
        for partition in ['normal','test']:verify_record(OUT/f'{name}_{partition}_extraction_protocol.json')
    validation=json.loads((OUT/'validation.json').read_text())
    assert validation['normal_files_verified']==888
    assert validation['full_and_holdout_normal_models_scores_reconstructed']==208
    assert validation['test_predictions_reconstructed']==528
    assert validation['probe_aurocs_independently_recomputed']==72
    assert validation['selection_normal_only'] and not validation['independent_recording_groups_available']
    evaluation=json.loads((OUT/'evaluation_audit.json').read_text());assert evaluation['control_predictions_exact_to_40']==330
    metrics=json.loads((OUT/'metrics.json').read_text());assert len(metrics['variants'])==32
    review=json.loads((OUT/'visual_review.json').read_text())
    assert review['inspected']==['encoder_comparison.png','alarm_tradeoffs.png'] and review['passed']
    for name,h in review['sha256'].items():assert sha(OUT/name)==h
    report=json.loads((OUT/'report_data.json').read_text())
    for p,h in report['source_sha256'].items():assert sha(p)==h
    capacity=json.loads((OUT/'capacity_diagnostics.json').read_text())
    assert capacity['normal_only'] and not capacity['refitted'] and not capacity['label_access']
    assert capacity['source_normal_checkpoint_sha256']==sha(OUT/'normal_models_checkpoint.json')
    assert capacity['script_sha256']==sha('scripts/diagnose_capacity45.py')
    for row in capacity['summary']:
        group=[r for r in capacity['rows'] if r['run']==row['run']]
        assert len(group)==row['spaces']==70
        assert sum(r['at_rank_cap'] for r in group)==row['at_rank_cap']
        assert sum(r['below_target_variance'] for r in group)==row['below_95pct']
    visibility=json.loads((OUT/'probe_visibility.json').read_text())
    assert visibility['normal_only'] and not visibility['selection_changed']
    assert visibility['probe_protocol_sha256']==sha(OUT/'probe_protocol.json')
    assert visibility['selection_sha256']==sha(OUT/'selection.json')
    assert visibility['script_sha256']==sha('scripts/diagnose_probe_visibility45.py')
    tests=json.loads((OUT/'tests.json').read_text());assert tests['exit_code']==0 and tests['passed']>=190
    files=[Path('README.md'),Path('src/ipad_vad/large_visual.py'),Path('docs/MODEL_SCALING_PROGRESS.md'),
           Path('docs/EXPERIMENT46_PLAN.md')]+list(Path('docs').glob('EXPERIMENT45*.md'))
    files+=list(Path('scripts').glob('*45*.py'))+list(Path('configs').glob('experiment45_*.json'))+list(OUT.rglob('*'))
    files=sorted(set(p for p in files if p.name!='public_manifest.json'))
    links=0
    for p in files:
        assert p.is_file(),p
        assert p.suffix in ['.py','.md','.json','.csv','.png'],p
        assert p.stat().st_size<50_000_000,p
        if p.suffix=='.json':json.loads(p.read_text())
        if p.suffix=='.md':
            text=p.read_text()
            if p.name=='README.md':text=text[text.index('## 실험 45'):]
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if '://' in target or target.startswith('#'):continue
                assert (p.parent/target.split('#')[0]).exists(),(p,target);links+=1
    table=(OUT/'summary_table.md').read_text().strip()
    assert table in Path('README.md').read_text() and table in Path('docs/EXPERIMENT45.md').read_text()
    write(OUT/'public_manifest.json',{'file_sha256':{str(p):sha(p) for p in files},'files':len(files),
        'links_checked':links,'raw_media_weights_features_logs_excluded':True,'protocols_verified':True,
        'plots_visually_inspected':review['inspected'],'tests_passed':tests['passed'],
        'goal_scope':'Only45 complete; larger-model learning and feature/temporal experiments remain.'})
    print('PUBLICATION READY',len(files),'files;',links,'links',flush=True)


if __name__=='__main__':main()
