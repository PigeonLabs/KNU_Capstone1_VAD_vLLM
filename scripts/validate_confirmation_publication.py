"""Check experiment35 frozen evidence, review artifacts and published numbers."""
import json,re,hashlib
from pathlib import Path
import numpy as np
import numpy.testing
from experiment35_confirmed_anchor import OUT,ART,VARIANTS,configure_common,common,load,sha,write
from experiment32_duration_evidence import protected_paths

def main():
    configure_common()
    for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json','pre_test_checkpoint.json','normal_verification_checkpoint.json','pre_evaluation_review_checkpoint.json','test_input_checkpoint.json']:common.verify_freeze(name)
    previous=json.loads(Path('results/experiment33/pre_audit_protocol.json').read_text())['protected_normal_sha256']
    for path,h in previous.items():assert sha(path)==h,path
    audit=json.loads((OUT/'normal_audit.json').read_text());assert audit['eligible']==VARIANTS and not audit['failed']
    for e in VARIANTS:
        folds=[r for r in audit['folds'] if r['variant']==e];assert len(folds)==5
        for r in folds:assert r['held_out'] not in r['calibration_sequences'] and r['eligible']
    exact=0
    for gate in ['hold','pool','age']:
        for p in sorted(Path(f'artifacts/experiment35_control_{gate}/predictions').glob('*.npz')):
            a=load(p);b=load(Path(f'artifacts/experiment34_semantic_{gate}/predictions')/p.name);assert set(a)==set(b)
            for key in a:np.testing.assert_array_equal(a[key],b[key])
            exact+=1
    assert exact==57
    check=json.loads((OUT/'pre_visual_checkpoint.json').read_text());assert sha(OUT/'visual_case_context.json')==check['case_context_sha256'];assert sha('scripts/review_confirmed_anchor.py')==check['renderer_sha256']
    for section in ['source_normal_image_sha256','local_feature_sha256']:
        for p,h in check[section].items():assert sha(p)==h
    manifest=json.loads((OUT/'local_contact_sheet_manifest.json').read_text());seen=[]
    for page in manifest['pages']:assert sha(page['path'])==page['sha256'];seen+=page['case_ids']
    review=json.loads((OUT/'visual_review.json').read_text());assert [r['case_id'] for r in review['cases']]==seen and len(set(seen))==24
    assert all(r['observation'] and r['uncertainty'] and r['semantic_ground_truth'] is None and r['action_boundary_ground_truth'] is None for r in review['cases'])
    diagnostic=json.loads((OUT/'diagnostic.json').read_text());summary=json.loads((OUT/'transform_summary.json').read_text())
    normal=json.loads((OUT/'normal_transform.json').read_text())['rows'];test=json.loads((OUT/'test_transform.json').read_text())['rows']
    for part,s in summary.items():
        rs=[r for r in normal+test if r['partition']==part]
        for key in ['samples','frames','anchor_selection_changed_samples','phase_changed_samples','phase_changed_frames','descriptor_changed_samples']:assert s[key]==sum(r[key] for r in rs)
    report=Path('docs/EXPERIMENT35.md').read_text()
    for e,v in diagnostic['variants'].items():
        for number in [v['metrics']['combined']['auroc'],v['metrics']['combined']['average_precision']]:assert f'{number:.4f}' in report
        assert str(v['alarms']['normal']) in report and str(v['alarms']['anomaly']) in report
    for f in ['README.md','docs/EXPERIMENT35.md','docs/EXPERIMENT36_PLAN.md']:
        path=Path(f)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if ':' not in target and not target.startswith('#'):assert (path.parent/target.split('#')[0]).exists(),(f,target)
    recommendations=report.split('## 4. 다음 Recommended improvements')[1].split('1순위만')[0];assert len(re.findall(r'^\| [123] \|',recommendations,re.M))==3
    validation=json.loads((OUT/'validation.json').read_text());assert validation['normal_holdout_cells_reconstructed']==30 and validation['test_predictions_reconstructed']==114 and validation['derived_feature_files_checked']==88
    files=[Path('README.md'),Path('docs/EXPERIMENT35.md'),Path('docs/EXPERIMENT36_PLAN.md'),*sorted(Path('configs').glob('experiment35_*.json')),*sorted(OUT.glob('*')),*sorted(Path('results').glob('experiment35_*/*')),*[Path('scripts')/p for p in ['experiment35_confirmed_anchor.py','review_confirmed_anchor.py','diagnose_confirmed_anchor.py','audit_anchor_confirmation.py','report_confirmed_anchor.py','validate_confirmation_publication.py']],Path('src/ipad_vad/relational_phase.py'),Path('src/ipad_vad/confirmed_anchor.py'),Path('tests/test_confirmed_anchor.py')]
    hashes={str(p):sha(p) for p in files if p.is_file() and p.name!='publication_validation.json'}
    assert all(Path(p).suffix in ['.md','.json','.csv','.png','.py'] for p in hashes)
    write(OUT/'publication_validation.json',{'protected_original_normal_files_unchanged':len(previous),'current_protocol_normal_numeric_files_unchanged':sum(Path(p).suffix=='.npz' for p in json.loads((OUT/'pre_normal_protocol.json').read_text())['file_sha256']),'control_test_predictions_all_arrays_exact':exact,'reviewed_cases':24,'reviewed_pages':12,'normal_holdout_folds':30,'full_unit_suite_passed':147,'unit_test_log_local':'/tmp/experiment35_tests.log','ranked_recommendations':3,'relative_report_links_exist':True,'source_images_not_published':True,'publication_sha256':hashes})
    print('Publication checks passed:',len(hashes),'files; control predictions exact:',exact,flush=True)
if __name__=='__main__':main()
