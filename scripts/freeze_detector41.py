"""Freeze normal annotation and training implementation before the actual run."""
import json
from pathlib import Path
from ipad_vad.learned_detector import sha,write


def main():
    output=Path('results/experiment41/detector_protocol.json')
    if output.exists():raise RuntimeError('Protocol exists; do not overwrite')
    files=[Path(p) for p in ['configs/experiment41_detector.json','results/experiment41/annotations.json','results/experiment41/annotation_review_decisions.json','results/experiment41/smoke.json','results/experiment41/initial_native_loss_smoke.json','results/experiment41/initial_candidate_inventory.json','results/experiment41/candidate_inventory_recomputed.json','results/experiment40/data_manifest.json','results/stage00/splits.json','src/ipad_vad/learned_detector.py','scripts/train_detector41.py','scripts/prepare_detector41_annotations.py','scripts/finalize_detector41_annotations.py','scripts/inventory_detector41_candidates.py','scripts/freeze_detector41.py','tests/test_learned_detector.py','docs/EXPERIMENT41_PLAN.md','artifacts/model_paths.json']]
    record=json.loads(Path('artifacts/model_paths.json').read_text())['IDEA-Research/grounding-dino-tiny']
    files+=sorted(Path(record['local_path']).glob('*'))
    for name in ['models/grounding_dino/modeling_grounding_dino.py','loss/loss_grounding_dino.py']:
        files.append(Path('.venv/lib/python3.12/site-packages/transformers')/name)
    smoke=json.loads(Path('results/experiment41/smoke.json').read_text());assert smoke['completed'] and smoke['frozen_hash_unchanged'] and smoke['checkpoint_restore_exact']
    write(output,{'stage':'before_shared_detector_training','normal_only':True,'file_sha256':{str(p):sha(p) for p in files if p.is_file()},'model_revision':record['revision'],'new_supervision':'reviewed_model_assisted_weak_boxes_not_human_ground_truth','annotations_train':178,'annotations_validation':50,'anomaly_labels_used':False,'calibration_or_test_used':False})
    print('Detector protocol frozen')

if __name__=='__main__':main()
