"""Verify experiment38 evidence and report consistency before publication."""
import csv
import json
import re
from pathlib import Path

import numpy as np

from experiment38_initial_observation import (
    OUT, SOURCE, VARIANTS, GATES, load, write, sha, configure_common, common,
)


def read(name):
    return json.loads((OUT / name).read_text())


def exact_files(left, right):
    paths = sorted(Path(left).glob('*.npz'))
    assert len(paths) == 19
    for path in paths:
        a, b = load(path), load(Path(right) / path.name)
        assert set(a) == set(b)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f'{path}:{key}')
    return len(paths)


def main():
    configure_common()
    for name in ['pre_normal_protocol.json', 'pre_calibration_checkpoint.json',
                 'pre_test_checkpoint.json', 'normal_verification_checkpoint.json',
                 'test_input_checkpoint.json']:
        common.verify_freeze(name)
    prior = json.loads(Path('results/experiment33/pre_audit_protocol.json').read_text())['protected_normal_sha256']
    for path, digest in prior.items():
        assert sha(path) == digest, path
    for name in ['prepare_access.json', 'normal_access.json']:
        access = read(name)
        assert access['test_data_opened'] is False
        assert all(not Path(p).name.startswith('testing_') and '/predictions/' not in p
                   for p in access['opened_normal_numeric_data'])
    audit = read('normal_audit.json')
    assert audit['normal_only'] and audit['eligible'] == VARIANTS and not audit['failed']
    assert len(audit['folds']) == 30
    assert all(r['eligible'] and r['held_out'] not in r['calibration_sequences'] for r in audit['folds'])
    routing = read('normal_routing_audit.json')
    assert len(routing['rows']) == 90
    assert routing['control_normal_models_and_scores_exact_to37'] == 3
    for key in ['used_request_cdf_pca_process_unchanged', 'post_first_scores_unchanged', 'pool_age_scores_unchanged']:
        assert routing[key]
    for name, changed in routing['allowed_array_changes'].items():
        assert set(changed) == ({'calibration_1', 'calibration_2', 'calibration_-1'} if name.startswith('hold_') else set())
    controls = sum(exact_files(f'artifacts/experiment38_control_{g}/predictions',
                              f'artifacts/experiment37_asinh_{g}/predictions') for g in GATES)
    negatives = sum(exact_files(f'artifacts/experiment38_guarded_{g}/predictions',
                               f'artifacts/experiment38_control_{g}/predictions') for g in ['pool', 'age'])
    validation = read('validation.json')
    for key, count in [('normal_holdout_cells_reconstructed', 30), ('test_predictions_reconstructed', 114),
                       ('source_feature_files_reused', 44), ('control_test_prediction_files_all_arrays_exact', controls),
                       ('pool_age_guarded_prediction_files_all_arrays_exact', negatives)]:
        assert validation[key] == count
    assert validation['scores_after_first_observation_exact']
    assert len(list(SOURCE.glob('*.npz'))) == 44
    context = read('initial_and_dwell_context.json')
    assert context['initial_mask_prefix_checks'] == 88 and context['source_features_checked'] == 44
    dwell = context['dwell_identity_summary']
    assert dwell['dwell_valid_frames'] - dwell['same_pair_since_entry_valid_frames'] == 144
    assert (dwell['dwell_only_fp_with_continuity'], dwell['dwell_only_tp_with_continuity']) == (13, 19)
    assert dwell['dwell_only_fp_without_continuity'] == dwell['dwell_only_tp_without_continuity'] == 0
    diagnostic = read('diagnostic.json')
    assert len({v['q99'] for v in diagnostic['variants'].values()}) == 1
    for gate in GATES:
        contrast = diagnostic['paired_contrasts'][gate]
        assert contrast['added'] == {'normal': 0, 'anomaly': 0}
        assert contrast['removed'] == {'normal': 60 if gate == 'hold' else 0, 'anomaly': 0}
        assert contrast['events'] == {'gained': [], 'lost': []}
        before = context['test_region_summary'][f'38_control_{gate}']
        after = context['test_region_summary'][f'38_guarded_{gate}']
        for key in ['initial_frames', 'fp_after', 'tp_after']:
            assert before[key] == after[key]
        assert before['initial_frames'] == 652 and after['fp_before'] == after['tp_before'] == 0
    assert all(r['initial_anomaly_frames'] == 0 for r in diagnostic['initial_region_per_sequence'])
    rows = list(csv.DictReader((OUT / 'comparison.csv').open()))
    assert len(rows) == 6
    report = Path('docs/EXPERIMENT38.md').read_text()
    readme = Path('README.md').read_text().split('## 실험 38 —')[1]
    for row in rows:
        variant = diagnostic['variants'][row['variant']]
        for col, key in [('combined_auroc', 'auroc'), ('combined_ap', 'average_precision')]:
            assert float(row[col]) == variant['metrics']['combined'][key]
            assert f'{float(row[col]):.4f}' in report and f'{float(row[col]):.4f}' in readme
        for col, key in [('fp', 'normal'), ('tp', 'anomaly')]:
            assert int(row[col]) == variant['alarms'][key]
    for name in ['README.md', 'docs/EXPERIMENT38.md', 'docs/EXPERIMENT39_PLAN.md']:
        path = Path(name)
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if ':' not in target and not target.startswith('#'):
                assert (path.parent / target.split('#')[0]).exists(), (name, target)
    assert len(re.findall(r'^\| [123] \|', report.split('## 4. 다음 Recommended improvements')[1], re.M)) == 3
    assert len(re.findall(r'^\| [123] \|', readme, re.M)) == 3
    testlog = Path('/tmp/experiment38_tests.log')
    assert '165 passed' in testlog.read_text()
    scripts = ['experiment38_initial_observation.py', 'diagnose_initial_observation.py',
               'audit_initial_observation_context.py', 'report_initial_observation.py',
               'validate_initial_observation_publication.py']
    files = [Path('README.md'), Path('docs/EXPERIMENT38.md'), Path('docs/EXPERIMENT39_PLAN.md'),
             *sorted(Path('configs').glob('experiment38_*.json')), *sorted(OUT.glob('*')),
             *sorted(Path('results').glob('experiment38_*/*')), *[Path('scripts') / s for s in scripts],
             Path('src/ipad_vad/scoring.py'), Path('src/ipad_vad/initial_observation.py'),
             Path('tests/test_initial_observation.py')]
    hashes = {str(p): sha(p) for p in files if p.is_file() and p.name != 'publication_validation.json'}
    assert all(Path(p).suffix in ['.json', '.md', '.csv', '.py', '.png'] for p in hashes)
    write(OUT / 'publication_validation.json', {
        'protected_original_normal_files_unchanged': len(prior),
        'current_protocol_normal_numeric_files_unchanged': sum(Path(p).suffix == '.npz' for p in read('pre_normal_protocol.json')['file_sha256']),
        'control_prediction_files_all_arrays_exact_to37': controls,
        'pool_age_negative_control_prediction_files_all_arrays_exact': negatives,
        'normal_routing_rows': 90, 'normal_holdout_cells': 30,
        'test_prediction_files_reconstructed': 114, 'source_features_reused': 44,
        'initial_mask_prefix_checks': 88, 'full_unit_suite_passed': 165,
        'unit_test_log_local': str(testlog), 'unit_test_log_sha256': sha(testlog),
        'ranked_recommendations': 3, 'relative_report_links_exist': True,
        'no_source_images_or_numeric_caches_published': True, 'publication_sha256': hashes,
    })
    print(f'Publication checks passed: {len(hashes)} files; {controls} historical and {negatives} negative-control prediction files exact', flush=True)


if __name__ == '__main__':
    main()
