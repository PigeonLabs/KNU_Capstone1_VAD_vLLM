"""Matched per-process models: change only vision features, retain experiment40 process."""
import copy,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.data import hold_scores
from ipad_vad.experiment import discovery_path
from experiment40_normal import arrays,factory
from candidates45_common import OUT,ART,CFG,config,guard,freeze,verify_record,sha,write

SCENES=config()['scenes'];RUNS=[*config()['controls'],*config()['candidates']]


def cfg(run,scene):return json.loads(Path(f'configs/experiment45_{run}_{scene}.json').read_text())
def load(run,scene,seq):
    with np.load(ART/run/'features'/scene/f'training_{seq}.npz',allow_pickle=False) as f:d=dict(f)
    d['sequence_id']=f'{scene}/training_{seq}';return d


def verify_normal():
    for name in ['pre_normal_models_protocol','normal_models_checkpoint','selection_checkpoint']:
        verify_record(OUT/f'{name}.json')


def setup():
    files=[]
    for run in RUNS:
        if run in config()['controls']:
            link=ART/run/'features';link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to(f'../../experiment40/{run}/features',target_is_directory=True)
            assert link.resolve()==Path(f'artifacts/experiment40/{run}/features').resolve()
        for scene in SCENES:
            options=json.loads(Path(f'configs/experiment40_A_{scene}.json').read_text())
            options.update(experiment=f'45_{run}_{scene}',scope=f'{scene}_larger_encoder_{run}',
                feature_source_experiment=f'45/{run}',representation_run=run,
                encoder=config()['candidates'][run]['repo'] if run in config()['candidates'] else f'experiment40/{run}',
                encoder_frozen=True)
            path=Path(f'configs/experiment45_{run}_{scene}.json')
            if path.exists():assert json.loads(path.read_text())==options
            else:write(path,options)
            files.append(path)
    return files


def main():
    guard();files=setup()
    if (OUT/'normal_models_checkpoint.json').exists():raise RuntimeError('Already frozen; verify instead of refitting')
    manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text())
    files.extend([Path(__file__),CFG,Path('scripts/experiment40_normal.py'),Path('scripts/candidates45_common.py'),
        Path('results/experiment40/data_manifest.json')])
    files.extend(sorted(Path('src/ipad_vad').glob('*.py')))
    files.extend(discovery_path(cfg('A',scene)) for scene in SCENES)
    originals=json.loads(Path('configs/experiment40_representation.json').read_text())
    for run in RUNS:
        record_path=(Path('results/experiment40')/f'{run}_normal_extraction.json' if run in config()['controls'] else OUT/f'{run}_normal_extraction.json')
        record=json.loads(record_path.read_text());assert record['normal_only'] and record['views']==manifest['view_count'];files.append(record_path)
        if run in config()['candidates']:verify_record(OUT/f'{run}_normal_extraction_protocol.json');files.append(OUT/f'{run}_model.json')
        for row in record['sequences']:
            path=ART/run/'features'/row['scene']/f'training_{row["sequence"]}.npz';assert sha(path)==row['sha256'];files.append(path)
            source=Path(originals['source_features'][row['scene']])/path.name
            assert sha(source)==manifest['normal_source_feature_sha256'][str(source)]
            with np.load(source,allow_pickle=False) as f:old=dict(f)
            with np.load(path,allow_pickle=False) as f:new=dict(f)
            assert set(new)==set(old)
            for key,value in new.items():
                if key in ['global_features','crop_features']:
                    assert len(value)==len(old[key]) and np.isfinite(value).all()
                    np.testing.assert_allclose(np.linalg.norm(value,axis=1),1,rtol=1e-5,atol=1e-6)
                else:np.testing.assert_array_equal(value,old[key])
    freeze(OUT/'pre_normal_models_protocol.json',files,normal_only=True)
    folds=[];full=[];protected=[];dest=ART/'normal_models';dest.mkdir(parents=True,exist_ok=True);exact40=0
    for scene in SCENES:
        split=manifest['subsplits'][scene];control={}
        for run in RUNS:
            fit=[load(run,scene,s) for s in split['downstream_fit']];cal={s:load(run,scene,s) for s in split['normal_calibration']}
            base=factory(cfg(run,scene));base.fit(fit)
            for held in [None,*cal]:
                m=copy.deepcopy(base);used=[s for s in cal if s!=held];m.calibrate([cal[s] for s in used])
                probe={s:cal[s] for s in cal if held is None or s==held};results={s:m.score(d) for s,d in probe.items()}
                keys=[k for k,v in next(iter(results.values())).items() if isinstance(v,np.ndarray)]
                saved={k:np.concatenate([r[k] for r in results.values()]) for k in keys};refs=arrays(m);suffix=f'{run}_{scene}_{held or "full"}'
                for kind,values in [('model',refs),('scores',saved)]:
                    p=dest/f'{suffix}_{kind}.npz';np.savez_compressed(p,**values);protected.append(p)
                    if run in config()['controls']:
                        with np.load(Path('artifacts/experiment40/normal_models')/p.name,allow_pickle=False) as f:
                            assert set(f)==set(values)
                            for k,v in values.items():np.testing.assert_array_equal(v,f[k])
                        exact40+=1
                params={k:v for k,v in refs.items() if k in ['transition','process_reference','allowed'] or k.startswith(('process_reference_state_','dwell_'))}
                process={k:saved[k] for k in keys if k not in ['visual','combined']}
                if run=='A':control[held]=(process,params)
                else:
                    for actual,expected in zip((process,params),control[held]):
                        assert set(actual)==set(expected)
                        for k,v in actual.items():np.testing.assert_array_equal(v,expected[k])
                finite=all(np.isfinite(saved[k]).all() and np.all((saved[k]>=0)&(saved[k]<=1)) for k in ['visual','process','combined'])
                alarms=sum(int(hold_scores(probe[s]['indices'],results[s]['combined']>m.threshold,int(probe[s]['frame_count'])).sum()) for s in probe)
                row={'run':run,'scene':scene,'held_out':held,'calibration_sequences':used,'q99':m.threshold,
                     'finite_bounded':bool(finite),'eligible':bool(finite and m.threshold<1),
                     'frames':sum(int(d['frame_count']) for d in probe.values()),'frame_alarms':alarms,'process_exact_to_A':True}
                if held is not None:assert held not in used;folds.append(row)
                else:
                    row['subspaces']=[{'role':r,'phase':p,'samples':s.n,'rank':s.rank} for (r,p),s in sorted(m.spaces.items())];full.append(row)
            print('Normal fit',scene,run,'q99',full[-1]['q99'],flush=True)
    eligible=[f'{r["run"]}_{r["scene"]}' for r in full if r['eligible'] and all(v['eligible'] for v in folds if v['run']==r['run'] and v['scene']==r['scene'])]
    write(OUT/'normal_models_audit.json',{'normal_only':True,'full':full,'folds':folds,'eligible':eligible,
        'control_arrays_exact_to_40':exact40,'note':'Per-calibration-video holdout for CDF/q99; identical process params/scores across encoders. Legacy fixed observation models may have seen current representation validation, as in40.'})
    freeze(OUT/'normal_models_checkpoint.json',protected+[OUT/'normal_models_audit.json',OUT/'pre_normal_models_protocol.json'],normal_only=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
