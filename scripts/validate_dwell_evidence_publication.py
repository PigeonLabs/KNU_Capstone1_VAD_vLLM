"""Verify experiment39 frozen evidence, paired results and publication contents."""
import csv,json,re
from pathlib import Path
import numpy as np
from experiment39_dwell_evidence import OUT,SOURCE,VARIANTS,GATES,load,write,sha,configure_common,common


def read(name):return json.loads((OUT/name).read_text())

def main():
    configure_common()
    for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json','pre_test_checkpoint.json','normal_verification_checkpoint.json','test_input_checkpoint.json']:common.verify_freeze(name)
    prior=json.loads(Path('results/experiment33/pre_audit_protocol.json').read_text())['protected_normal_sha256']
    for p,h in prior.items():assert sha(p)==h,p
    for name in ['prepare_access.json','normal_access.json']:
        a=read(name);assert a['test_data_opened'] is False
        assert all(not Path(p).name.startswith('testing_') and '/predictions/' not in p for p in a['opened_normal_numeric_data'])
    normal=read('normal_audit.json');assert normal['normal_only'] and normal['eligible']==VARIANTS and not normal['failed']
    assert len(normal['folds'])==30 and all(r['eligible'] and r['held_out'] not in r['calibration_sequences'] for r in normal['folds'])
    fusion=read('normal_fusion_audit.json');assert len(fusion['rows'])==90
    assert fusion['control_normal_models_and_prior_score_keys_exact_to38']==3
    assert fusion['all_non_threshold_model_arrays_unchanged'] and fusion['raw_scores_unchanged'] and fusion['gated_process_and_combined_nonincreasing']
    assert len(fusion['threshold_comparisons'])==18 and not any(r['threshold_changed'] for r in fusion['threshold_comparisons'].values())
    assert read('normal_evidence.json')['prefix_checks']==50
    count=0
    for gate in GATES:
        paths=sorted(Path(f'artifacts/experiment39_control_{gate}/predictions').glob('*.npz'));assert len(paths)==19
        for p in paths:
            a,b=load(p),load(Path(f'artifacts/experiment38_guarded_{gate}/predictions')/p.name)
            assert set(a)==set(b)|{'dwell_evidence_valid','dwell_gated'}
            for key,value in b.items():np.testing.assert_array_equal(a[key],value,err_msg=f'{p}:{key}')
            count+=1
    validation=read('validation.json')
    for key,value in [('normal_holdout_cells_reconstructed',30),('test_predictions_reconstructed',114),('source_feature_files_reused',44),('control_test_prediction_files_all_prior_arrays_exact',57),('test_mask_prefix_checks',38)]:assert validation[key]==value
    assert validation['raw_visual_transition_dwell_arrays_exact'] and validation['all_non_threshold_model_arrays_exact'] and validation['gated_scores_nonincreasing']
    assert len(list(SOURCE.glob('*.npz')))==44
    context=read('duration_and_gate_context.json');assert context['episode_parser_mask_equality_videos']==44 and context['no_distribution_refit']
    fit=context['normal_duration_support']['fit']
    assert [(fit[k]['legacy_complete'],fit[k]['same_pair_complete']) for k in ['0->1','1->2','2->1']]==[(13,9),(16,4),(7,4)]
    assert all(not r['strict_meets_minimum10'] for r in fit.values())
    assert [fit[k]['same_pair_right_censored'] for k in ['0->1','1->2','2->1']]==[10,22,2]
    diagnostic=read('diagnostic.json');assert len({v['q99'] for v in diagnostic['variants'].values()})==1
    for gate in GATES:
        contrast=diagnostic['paired_contrasts'][gate]
        assert contrast['added']==contrast['removed']=={'normal':0,'anomaly':0}
        assert contrast['events']=={'gained':[],'lost':[]}
        assert contrast['changed_combined_frames']==24 and contrast['changed_process_frames']==76
        a,b=[diagnostic['variants'][f'39_{g}_{gate}'] for g in ['control','gated']]
        assert a['alarms']==b['alarms'] and a['events']==b['events']
        assert a['dwell_valid_frames']==b['dwell_valid_frames']==1889 and b['dwell_eligible_frames']==1745
        for metric in ['auroc','average_precision']:assert b['metrics']['combined'][metric]<a['metrics']['combined'][metric]
        for g in ['control','gated']:
            branch=diagnostic['branch_contribution'][f'39_{g}_{gate}'];assert branch['dwell_only_alarms']=={'normal':13,'anomaly':19} and branch['process_added_events']=={'gained':[],'lost':[]}
        effects=context['test_gate_effect_summary'][gate]
        assert effects['blocked_frames']==144 and effects['combined_changed_normal']==4 and effects['combined_changed_anomaly']==20
    rows=list(csv.DictReader((OUT/'comparison.csv').open()));assert len(rows)==6
    report=Path('docs/EXPERIMENT39.md').read_text();readme=Path('README.md').read_text().split('## 실험 39 —')[1]
    for row in rows:
        v=diagnostic['variants'][row['variant']]
        for col,key in [('combined_auroc','auroc'),('combined_ap','average_precision')]:
            assert float(row[col])==v['metrics']['combined'][key]
            assert f'{float(row[col]):.4f}' in report and f'{float(row[col]):.4f}' in readme
        for col,key in [('fp','normal'),('tp','anomaly')]:assert int(row[col])==v['alarms'][key]
    for name in ['README.md','docs/EXPERIMENT39.md','docs/EXPERIMENT40_PLAN.md']:
        path=Path(name)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if ':' not in target and not target.startswith('#'):assert (path.parent/target.split('#')[0]).exists(),(name,target)
    assert len(re.findall(r'^\| [123] \|',report.split('## 4. 다음 Recommended improvements')[1],re.M))==3
    assert len(re.findall(r'^\| [123] \|',readme,re.M))==3
    testlog=Path('/tmp/experiment39_tests.log');assert '173 passed' in testlog.read_text()
    scripts=['experiment39_dwell_evidence.py','diagnose_dwell_evidence.py','audit_dwell_evidence_context.py','report_dwell_evidence.py','validate_dwell_evidence_publication.py','evaluate_baseline.py']
    files=[Path('README.md'),Path('docs/EXPERIMENT39.md'),Path('docs/EXPERIMENT40_PLAN.md'),*sorted(Path('configs').glob('experiment39_*.json')),*sorted(OUT.glob('*')),*sorted(Path('results').glob('experiment39_*/*')),*[Path('scripts')/s for s in scripts],Path('src/ipad_vad/dwell_scoring.py'),Path('src/ipad_vad/dwell_evidence.py'),Path('tests/test_dwell_evidence.py')]
    hashes={str(p):sha(p) for p in files if p.is_file() and p.name!='publication_validation.json'}
    assert all(Path(p).suffix in ['.json','.md','.csv','.py','.png'] for p in hashes)
    write(OUT/'publication_validation.json',{'protected_original_normal_files_unchanged':len(prior),'current_protocol_normal_numeric_files_unchanged':sum(Path(p).suffix=='.npz' for p in read('pre_normal_protocol.json')['file_sha256']),'control_prediction_files_all_prior_arrays_exact_to38':count,'new_prediction_diagnostic_keys':['dwell_evidence_valid','dwell_gated'],'normal_fusion_rows':90,'normal_holdout_cells':30,'test_prediction_files_reconstructed':114,'source_features_reused':44,'normal_and_test_prefix_checks':88,'episode_parser_mask_equality_videos':44,'full_unit_suite_passed':173,'unit_test_log_local':str(testlog),'unit_test_log_sha256':sha(testlog),'ranked_recommendations':3,'relative_report_links_exist':True,'no_source_images_or_numeric_caches_published':True,'publication_sha256':hashes})
    print('Publication checks passed:',len(hashes),'files;',count,'control files preserve all prior arrays; added diagnostics explicit',flush=True)

if __name__=='__main__':main()
