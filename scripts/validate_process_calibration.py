"""Check experiment 09 changes calibration only, against saved experiment 08."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from ipad_vad.data import hold_scores
from ipad_vad.scoring import empirical_percentile
from ipad_vad.experiment import load_process


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',default='08');p.add_argument('--experiment',default='09');args=p.parse_args()
    cfg=json.loads(Path(f'configs/experiment{args.experiment}.json').read_text());scene=cfg['scene'];process=load_process(cfg)
    source=Path(f'artifacts/experiment{args.baseline}');target=Path(f'artifacts/experiment{args.experiment}')
    before=load(source/'normal_model.npz');after=load(target/'normal_model.npz')
    assert set(after)-set(before)=={f'process_reference_state_{i}' for i in range(len(process['phases']))}
    for key in before:
        if key!='threshold':np.testing.assert_array_equal(before[key],after[key],err_msg=key)
    protocol=json.loads(Path(f'results/experiment{args.experiment}/pre_evaluation_protocol.json').read_text())
    assert protocol['config_sha256']==hashlib.sha256(Path(f'configs/experiment{args.experiment}.json').read_bytes()).hexdigest()
    k=len(process['phases']);allowed=np.eye(k,dtype=bool);ids={v['id']:i for i,v in enumerate(process['phases'])};order=[ids[v] for v in process['normal_order']]
    for a,b in zip(order[:-1],order[1:]):allowed[a,b]=True
    if process['cyclic']:allowed[order[-1],order[0]]=True
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    rows=[];calraw={i:[] for i in range(k)};n=0
    for path in sorted((source/'features'/scene).glob('*.npz')):
        dest=target/'features'/scene/path.name
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==protocol['source_sha256'][path.name]
        assert path.read_bytes()==dest.read_bytes();n+=1
        d=load(dest);phases=d['phases'];a,b=phases[:-1],phases[1:]
        raw=np.r_[0.,-np.log(after['transition'][a,b])+(~allowed[a,b]).astype(float)]
        if path.stem.startswith('training_'):
            if path.stem.split('_')[1] in split['calibration']:
                for i in range(k):calraw[i].extend(raw[1:][a==i].tolist())
            continue
        name=f'{scene}_{path.stem.split("_")[1]}.npz';old=load(source/'predictions'/name);new=load(target/'predictions'/name)
        for key in old:
            if key not in ('process','combined'):np.testing.assert_array_equal(old[key],new[key],err_msg=name+':'+key)
        global_score=empirical_percentile(after['process_reference'],raw);expected=global_score.copy();conditioned=np.zeros(len(raw),bool)
        for i in range(k):
            ref=after[f'process_reference_state_{i}']
            if len(ref)>=cfg['process_calibration']['minimum_support']:
                positions=np.flatnonzero(a==i)+1;conditioned[positions]=True
                expected[positions]=empirical_percentile(ref,raw[positions])
        np.testing.assert_array_equal(old['process'],hold_scores(d['indices'],global_score,len(old['labels'])))
        np.testing.assert_array_equal(new['process'],hold_scores(d['indices'],expected,len(new['labels'])))
        np.testing.assert_array_equal(new['process_conditioned'],conditioned)
        w=cfg['visual_process_weight'];np.testing.assert_array_equal(new['combined'],w*new['visual']+(1-w)*new['process'])
        assert np.isfinite(new['combined']).all()
        rows.append({'sequence':name,'sampled':len(raw),'conditional':int(conditioned.sum()),'first_sample_fallback':1,'insufficient_support_fallback':int((~conditioned[1:]).sum())})
    for i,values in calraw.items():np.testing.assert_array_equal(after[f'process_reference_state_{i}'],values)
    out={'feature_sequences_checked':n,'test_predictions_checked':len(rows),'shared_normal_model_arrays_except_threshold_identical':True,
         'all_features_phases_visual_objects_and_labels_identical':True,'old_global_and_new_conditional_scores_reconstructed':True,
         'conditional_references_from_normal_calibration_only':True,'config_matches_pre_evaluation_freeze':True,
         'sequences':rows,'limitations':'Numerical and provenance checks; no semantic-state or localization accuracy claim.'}
    Path(f'results/experiment{args.experiment}/validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
