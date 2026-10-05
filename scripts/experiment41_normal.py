"""Refit per-process statistical models after the one shared detector change."""
import copy,json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.data import hold_scores
from ipad_vad.learned_detector import sha,write
from experiment40_normal import SCENES,RUNS,factory as strict_factory,arrays
from experiment41_phases import load_npz,verify_control

OUT=Path('results/experiment41');ART=Path('artifacts/experiment41')


def cfg(run,scene):return json.loads(Path(f'configs/experiment41_{run}_{scene}.json').read_text())
def factory(options):
    if options.get('dwell_allow_unavailable'):
        from ipad_vad.optional_dwell import OptionalLognormalDwellBaseline
        from ipad_vad.experiment import load_process
        return OptionalLognormalDwellBaseline(options,load_process(options))
    return strict_factory(options)
def load(run,scene,seq):
    d=load_npz(ART/run/'features'/scene/f'training_{seq}.npz');d['sequence_id']=f'{scene}/training_{seq}';return d


def setup():
    files=[]
    for scene in SCENES:
        for run in RUNS:
            options=json.loads(Path(f'configs/experiment40_{run}_{scene}.json').read_text());options.update(experiment=f'41_{run}_{scene}',feature_source_experiment=f'41/{run}',scope=f'{scene}_shared_learned_detector_frozen_visual_{run}',encoder_frozen=True,representation_checkpoint_experiment='40')
            if scene=='R04':options['dwell_allow_unavailable']=True
            path=Path(f'configs/experiment41_{run}_{scene}.json')
            if path.exists():assert json.loads(path.read_text())==options
            else:write(path,options)
            files.append(path)
    return files


def verify_normal():
    for name in ['detector_protocol','pre_normal_models_protocol','normal_models_checkpoint']:
        for path,h in json.loads((OUT/f'{name}.json').read_text())['file_sha256'].items():assert sha(path)==h,path


def main():
    if (OUT/'normal_models_checkpoint.json').exists():raise RuntimeError('Completed normal stage exists')
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            s=str(args[0])
            if '/test_label/' in s or '/testing/' in s or Path(s).name.startswith('testing_'):raise RuntimeError('Test access forbidden during normal fitting')
    sys.addaudithook(guard)
    configs=setup();manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text());files=[*configs,Path(__file__),OUT/'detector_protocol.json',OUT/'training.json',OUT/'detector_training_extraction.json',Path('results/experiment40/data_manifest.json')]
    files+=sorted(Path('src/ipad_vad').glob('*.py'))
    files += [Path('tests/test_optional_dwell.py'),Path('docs/EXPERIMENT41_NORMAL_RECOVERY.md'),OUT/'normal_fit_recovery.json']
    files += [Path('artifacts/experiment41/detector/best.pt'),Path('artifacts/experiment08/relation_model.npz'),Path('artifacts/experiment17/anchor_text_features.npz'),Path('artifacts/experiment18/normal_relation_model.npz'),Path('artifacts/experiment36/normal_relation_model.npz'),Path('artifacts/experiment37/normal_relation_model.npz'),Path('results/experiment02/phase_grounding.json')]
    files += [Path(f'configs/experiment{e}.json') for e in ['02','08','18','34_semantic_hold','36_refit_hold']]
    files += [Path(f'scripts/{n}.py') for n in ['experiment35_confirmed_anchor','experiment36_phase_refit','experiment37_asinh_phase','audit_transition_provenance']]
    files += [Path(f'artifacts/experiment40/{run}/best.pt') for run in RUNS if run not in ['A','B']]
    files += [Path(cfg('A',scene)['process_discovery']) if 'process_discovery' in cfg('A',scene) else Path('results/experiment01/process_discovery.json') for scene in SCENES]
    files+=[Path('scripts')/n for n in ['extract_detector41.py','extract_detector41_representations.py','experiment41_phases.py','evaluate_detector41.py','validate_detector41.py','prepare_detector41_test.py','diagnose_detector41.py','experiment40_normal.py']]
    for p,h in json.loads((OUT/'detector_protocol.json').read_text())['file_sha256'].items():assert sha(p)==h,p
    assert verify_control()==111
    for run in RUNS:
        rec=OUT/f'{run}_training_extraction.json';files.append(rec);record=json.loads(rec.read_text());assert record['normal_only'] and not record['encoders_retrained']
        for row in record['sequences']:
            path=ART/run/'features'/row['scene']/f'training_{row["sequence"]}.npz';assert sha(path)==row['sha256'];files.append(path)
            original=load_npz(ART/'detections/features'/row['scene']/path.name);current=load_npz(path);assert set(original)==set(current)
            for k,v in original.items():
                if k not in ['global_features','crop_features']:np.testing.assert_array_equal(v,current[k])
            for k in ['global_features','crop_features']:
                assert current[k].shape==original[k].shape and np.isfinite(current[k]).all();np.testing.assert_allclose(np.linalg.norm(current[k],axis=1),1,rtol=1e-5,atol=1e-6)
    protocol={'normal_only':True,'file_sha256':{str(p):sha(p) for p in files}};write(OUT/'pre_normal_models_protocol.json',protocol)
    rows=[];full=[];dest=ART/'normal_models';dest.mkdir(parents=True,exist_ok=True);protected=[]
    for scene in SCENES:
        split=manifest['subsplits'][scene];control={}
        for run in RUNS:
            fit=[load(run,scene,s) for s in split['downstream_fit']];cal={s:load(run,scene,s) for s in split['normal_calibration']};base=factory(cfg(run,scene));base.fit(fit)
            for held in [None,*cal]:
                m=copy.deepcopy(base);used=[s for s in cal if s!=held];m.calibrate([cal[s] for s in used]);probe={s:cal[s] for s in cal if held is None or s==held};results={s:m.score(d) for s,d in probe.items()};keys=[k for k,v in next(iter(results.values())).items() if isinstance(v,np.ndarray)];saved={k:np.concatenate([r[k] for r in results.values()]) for k in keys};suffix=f'{run}_{scene}_{held or "full"}';references=arrays(m)
                for kind,values in [('model',references),('scores',saved)]:
                    path=dest/f'{suffix}_{kind}.npz';np.savez_compressed(path,**values);protected.append(path)
                process_keys=[k for k in keys if k not in ['visual','combined']];params={k:v for k,v in references.items() if k in ['transition','process_reference','allowed'] or k.startswith('process_reference_state_') or k.startswith('dwell_')}
                if run=='A':control[held]=({k:saved[k] for k in process_keys},params)
                else:
                    assert set(process_keys)==set(control[held][0])
                    for k in process_keys:np.testing.assert_array_equal(saved[k],control[held][0][k])
                    for k,v in params.items():np.testing.assert_array_equal(v,control[held][1][k])
                finite=all(np.isfinite(saved[k]).all() and np.all((saved[k]>=0)&(saved[k]<=1)) for k in ['visual','process','combined']);alarms=sum(int(hold_scores(probe[s]['indices'],results[s]['combined']>m.threshold,int(probe[s]['frame_count'])).sum()) for s in probe);count=sum(int(d['frame_count']) for d in probe.values())
                row={'run':run,'scene':scene,'held_out':held,'calibration_sequences':used,'q99':m.threshold,'finite_bounded':bool(finite),'eligible':bool(finite and m.threshold<1),'frames':count,'frame_alarms':alarms,'process_exact_to_A':True}
                if held is not None:assert held not in used;rows.append(row)
                else:
                    row['subspaces']=[{'role':r,'phase':p,'samples':v.n,'rank':v.rank} for (r,p),v in sorted(m.spaces.items())];row['dwell_context_support']=getattr(getattr(m,'dwell',None),'context_support',None);row['dwell_available']=getattr(getattr(m,'dwell',None),'available',None);row['dwell_unavailable_reason']=getattr(getattr(m,'dwell',None),'unavailable_reason',None);full.append(row)
            print(scene,run,'normal q99',full[-1]['q99'],flush=True)
    eligible=[f'{r["run"]}_{r["scene"]}' for r in full if r['eligible'] and all(v['eligible'] for v in rows if v['run']==r['run'] and v['scene']==r['scene'])]
    write(OUT/'normal_models_audit.json',{'normal_only':True,'full':full,'folds':rows,'eligible':eligible,'note':'Same original downstream FIT/calibration and normal-video holdout; all8 encoders frozen; changed detector observations propagate into refitted statistical references. Process arrays remain exact across visual arms.'})
    protected+=[OUT/'pre_normal_models_protocol.json',OUT/'normal_models_audit.json'];write(OUT/'normal_models_checkpoint.json',{'file_sha256':{str(p):sha(p) for p in protected},'normal_only':True});print('Normal models frozen',len(full),len(rows),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
