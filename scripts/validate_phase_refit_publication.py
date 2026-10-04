"""Verify experiment36 evidence and document links before publication."""
import csv,hashlib,json,re
from pathlib import Path
import numpy as np
from experiment36_phase_refit import OUT,ART,SOURCE,SPLIT,GROUPS,VARIANTS,load,write,sha,configure_common,common


def main():
    configure_common()
    for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json','pre_test_checkpoint.json','normal_verification_checkpoint.json','pre_evaluation_review_checkpoint.json','test_input_checkpoint.json']:common.verify_freeze(name)
    prior=json.loads(Path('results/experiment33/pre_audit_protocol.json').read_text())['protected_normal_sha256']
    for p,h in prior.items():assert sha(p)==h,p
    fit=json.loads((OUT/'phase_fit.json').read_text());access=json.loads((OUT/'phase_fit_access.json').read_text())
    assert fit['normal_fit_sequences']==SPLIT['fit'] and fit['calibration_sequences_excluded']==SPLIT['calibration']
    opened={Path(p).name for p in access['opened_numeric_during_phase_fit'] if Path(p).name.startswith('training_')};assert opened=={f'training_{s}.npz' for s in SPLIT['fit']}
    assert not fit['mapping_used_for_inference']
    phase=json.loads((OUT/'normal_phase_audit.json').read_text());assert phase['fit_model_exactly_reconstructed'] and phase['prefix_checks']==100
    review=json.loads((OUT/'fixed_case_phase_review.json').read_text());assert review['case_count']==24 and review['frames']==72 and not review['new_images_reviewed']
    assert sha(review['case_source'])==review['case_source_sha256']
    assert len({c['case']['case_id'] for c in review['cases']})==24
    for case in review['cases']:
        assert case['semantic_ground_truth'] is None and case['action_boundary_ground_truth'] is None
        for frame in case['frames']:assert frame['selection_unchanged']
    audit=json.loads((OUT/'normal_audit.json').read_text());assert audit['eligible']==VARIANTS and not audit['failed']
    assert len(audit['folds'])==30 and all(r['eligible'] and r['held_out'] not in r['calibration_sequences'] for r in audit['folds'])
    assert json.loads((OUT/'fit_rank_control.json').read_text())['control_rank_map_matches_35']
    count=0
    for gate in ['hold','pool','age']:
        for p in sorted(Path(f'artifacts/experiment36_control_{gate}/predictions').glob('*.npz')):
            a=load(p);b=load(Path(f'artifacts/experiment35_confirmed_{gate}/predictions')/p.name);assert set(a)==set(b)
            for k in a:np.testing.assert_array_equal(a[k],b[k])
            count+=1
    assert count==57
    validation=json.loads((OUT/'validation.json').read_text());assert validation['normal_holdout_cells_reconstructed']==30 and validation['test_predictions_reconstructed']==114 and validation['derived_feature_files_checked']==88
    diagnostic=json.loads((OUT/'diagnostic.json').read_text());summary=json.loads((OUT/'transform_summary.json').read_text());raw=json.loads((OUT/'normal_transform.json').read_text())['rows']+json.loads((OUT/'test_transform.json').read_text())['rows']
    for part,totals in summary.items():
        rows=[r for r in raw if r['partition']==part]
        for k in ['samples','frames','anchor_selection_changed_samples','phase_changed_samples','phase_changed_frames','descriptor_changed_samples']:assert totals[k]==sum(r[k] for r in rows)
        assert totals['anchor_selection_changed_samples']==totals['descriptor_changed_samples']==0
    csvrows=list(csv.DictReader((OUT/'comparison.csv').open()));assert len(csvrows)==6
    report=Path('docs/EXPERIMENT36.md').read_text();readme=Path('README.md').read_text().split('## 실험 36 —')[1]
    for row in csvrows:
        v=diagnostic['variants'][row['variant']]
        for col,key in [('combined_auroc','auroc'),('combined_ap','average_precision')]:
            assert float(row[col])==v['metrics']['combined'][key]
            assert f'{float(row[col]):.4f}' in report and f'{float(row[col]):.4f}' in readme
        for col,key in [('fp','normal'),('tp','anomaly')]:assert int(row[col])==v['alarms'][key]
    for f in ['README.md','docs/EXPERIMENT36.md','docs/EXPERIMENT37_PLAN.md']:
        path=Path(f)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if ':' not in target and not target.startswith('#'):assert (path.parent/target.split('#')[0]).exists(),(f,target)
    assert len(re.findall(r'^\| [123] \|',report.split('## 4. 다음 Recommended improvements')[1],re.M))==3
    assert len(re.findall(r'^\| [123] \|',readme,re.M))==3
    testlog=Path('/tmp/experiment36_tests.log');assert '152 passed' in testlog.read_text()
    scripts=['experiment36_phase_refit.py','audit_phase_refit.py','diagnose_phase_refit.py','audit_refit_duration_provenance.py','audit_refit_fallback.py','report_phase_refit.py','validate_phase_refit_publication.py']
    files=[Path('README.md'),Path('docs/EXPERIMENT36.md'),Path('docs/EXPERIMENT37_PLAN.md'),*sorted(Path('configs').glob('experiment36_*.json')),*sorted(OUT.glob('*')),*sorted(Path('results').glob('experiment36_*/*')),*[Path('scripts')/s for s in scripts],Path('src/ipad_vad/phase_refit.py'),Path('tests/test_phase_refit.py')]
    hashes={str(p):sha(p) for p in files if p.is_file() and p.name!='publication_validation.json'}
    assert all(Path(p).suffix in ['.json','.md','.csv','.py','.png'] for p in hashes)
    write(OUT/'publication_validation.json',{'protected_original_normal_files_unchanged':len(prior),'current_protocol_normal_numeric_files_unchanged':sum(Path(p).suffix=='.npz' for p in json.loads((OUT/'pre_normal_protocol.json').read_text())['file_sha256']),'normal_fit_cache_files':len(opened),'normal_fit_calibration_excluded':True,'normal_fit_recipe_independently_reconstructed':True,'control_test_predictions_all_arrays_exact':count,'same_fixed_cases_metadata_only':24,'fixed_case_frames_selection_unchanged':72,'normal_prefix_checks':100,'normal_holdout_cells':30,'test_prediction_files_reconstructed':114,'derived_feature_files_checked':88,'full_unit_suite_passed':152,'unit_test_log_local':str(testlog),'unit_test_log_sha256':sha(testlog),'ranked_recommendations':3,'relative_report_links_exist':True,'no_source_images_or_numeric_caches_published':True,'publication_sha256':hashes})
    print('Publication checks passed:',len(hashes),'files;',count,'control prediction files exact',flush=True)

if __name__=='__main__':main()
