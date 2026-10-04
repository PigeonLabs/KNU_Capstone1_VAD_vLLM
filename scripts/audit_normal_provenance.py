"""Audit experiment13's normal-video feature lineage and FIT-only relation model."""
import hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.relational_phase import RelationalPhase
from ipad_vad.normal_validation import leave_one_video_out


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    cfg=json.loads(Path('configs/experiment13.json').read_text());scene=cfg['scene'];split=json.loads(Path(cfg['split_source']).read_text())[scene]
    folds=leave_one_video_out(split['fit'],split['calibration']);record=json.loads(Path('results/experiment07/process_discovery.json').read_text())
    discovery_ids=sorted({Path(s).parts[3] for s in record['sources']})
    assert set(discovery_ids)<=set(split['fit']) and not set(discovery_ids)&set(split['calibration'])
    phasecfg=json.loads(Path('configs/experiment08.json').read_text());assert phasecfg['encoder_frozen']
    current=Path(f'artifacts/experiment{cfg["source_features_experiment"]}/features/{scene}');source=Path(f'artifacts/experiment07/features/{scene}')
    ids=split['fit']+split['calibration'];data={s:load(current/f'training_{s}.npz') for s in ids};hashes={}
    for seq in ids:
        original=load(source/f'training_{seq}.npz')
        for key in original:
            if key!='phases':np.testing.assert_array_equal(original[key],data[seq][key],err_msg=seq+':'+key)
        for folder in [source,current]:
            path=folder/f'training_{seq}.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    model=RelationalPhase(**phasecfg['relational_phase'])
    with threadpool_limits(limits=4):model.fit([data[s] for s in split['fit']])
    saved=load(Path('artifacts/experiment08/relation_model.npz'))
    differences={}
    for key,values in [('location',model.location),('scale',model.scale),('centers',model.centers)]:
        differences[key]=float(np.max(np.abs(values-saved[key])))
        np.testing.assert_allclose(values,saved[key],rtol=0,atol=1e-12)
    for role,value in zip(saved['area_roles'],saved['area_upper']):assert model.area_upper[int(role)]==value
    for seq,d in data.items():
        phase,valid,selected,x=model.transform(d)
        for key,value in [('phases',phase),('relation_valid',valid),('relation_detection_indices',selected),('relation_descriptors',x)]:np.testing.assert_array_equal(d[key],value,err_msg=seq+':'+key)
    out={'scene':scene,'fit_sequences':split['fit'],'normal_calibration_pool':split['calibration'],'folds':folds,'discovery_fit_sequences':discovery_ids,
         'discovery_excludes_calibration_pool':True,'normal_feature_sequences_checked':len(ids),'appearance_detection_tracking_arrays_match_experiment07':True,
         'relation_model_reconstructed_from_fit_only_within_tolerance':True,'relation_parameter_max_abs_difference':differences,'relation_parameter_absolute_tolerance':1e-12,'all_normal_phase_assignments_reproduced':True,'encoder_frozen_config':True,
         'source_sha256':hashes,'limitations':'Checks cached feature lineage and train-only relation reconstruction; does not prove original recording-group independence or pretraining-data independence.'}
    Path('results/experiment13/provenance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
