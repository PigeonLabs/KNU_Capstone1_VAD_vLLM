"""Experiment 02: replace ONLY phase assignments, reuse exact experiment 01 features."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.spatial_phase import SpatialPhase


def main():
    cfg=json.loads(Path('configs/experiment02.json').read_text());scene=cfg['scene']
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    source=Path('artifacts/experiment01/features')/scene;target=Path('artifacts/experiment02/features')/scene
    target.mkdir(parents=True,exist_ok=True);out=Path('results/experiment02');out.mkdir(parents=True,exist_ok=True)
    def load(path):
        with np.load(path,allow_pickle=False) as f:return dict(f)
    grounding=SpatialPhase(**cfg['spatial_phase'])
    with threadpool_limits(limits=4):grounding.fit([load(source/f'training_{s}.npz') for s in split['fit']])
    diagnostics=[]
    for path in sorted(source.glob('*.npz')):
        data=load(path);old=data['phases'].copy()
        data['phases'],observed,chosen,positions=grounding.transform(data)
        np.savez_compressed(target/path.name,**data)
        # Verify this ablation changes no detector output or encoder feature.
        check=load(target/path.name);original=load(path)
        for key in original:
            if key!='phases' and not np.array_equal(check[key],original[key]):raise ValueError(f'Unintended feature change: {key}')
        diagnostics.append({'sequence_key':path.stem,'sampled_frames':len(old),'direct_bbox_observations':int(observed.sum()),
                            'held_phase_samples':int((~observed).sum()),'phase_changed_samples':int((old!=data['phases']).sum()),
                            'phase_counts':np.bincount(data['phases'],minlength=grounding.k).tolist()})
    result={'experiment':'02','motivation':'Experiment 01 has zero FIT samples in semantic middle phase; replace global CLIP phase lookup with normal-track spatial grounding.',
            'fit_sequences':split['fit'],'parameters':cfg['spatial_phase'],'grounding_model':grounding.evidence,'sequences':diagnostics,
            'only_phases_changed':True,'test_labels_used_for_grounding':False,
            'missing_policy':'Hold prior phase; before first observation use phase 0. Smoothing uses last three valid observations, which may be stale across missing intervals.',
            'limitations':['R01-specific increasing spatial phase order.','Normal detector track errors may contaminate path estimation.',
                           'Missing detections remain unsolved and are explicitly counted.','Not validation of semantic phase accuracy.']}
    (out/'phase_grounding.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['grounding_model'],indent=2))


if __name__=='__main__':main()
