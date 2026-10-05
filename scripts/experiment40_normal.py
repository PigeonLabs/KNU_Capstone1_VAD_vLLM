"""Separate normal models/calibration for each process after encoder selection."""
import copy,json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.scoring import Baseline
from ipad_vad.completed_dwell import CompletedDwellBaseline
from ipad_vad.lognormal_dwell import LognormalDwellBaseline
from ipad_vad.experiment import load_process,discovery_path
from ipad_vad.data import hold_scores
from prepare_representation40 import OUT,ART,sha,write

SCENES=['R01','R02','R03','R04'];RUNS=['A','B',*[f'{a}_s{s}' for s in [42,43,44] for a in ['C','D']]]
TEMPLATES={'R01':'02','R02':'40_R02_base','R03':'14','R04':'39_gated_hold'}

def cfg(run,scene):return json.loads(Path(f'configs/experiment40_{run}_{scene}.json').read_text())
def factory(options):
    cls=LognormalDwellBaseline if options.get('dwell_score')=='fit_lognormal_cdf' else CompletedDwellBaseline if options.get('dwell_score')=='fit_complete_percentile' else Baseline
    return cls(options,load_process(options))
def load(run,scene,seq):
    with np.load(ART/run/'features'/scene/f'training_{seq}.npz',allow_pickle=False) as f:d=dict(f)
    d['sequence_id']=f'{scene}/training_{seq}';return d

def arrays(m):
    out={'threshold':np.array(m.threshold),'transition':m.transition,'process_reference':m.process_reference,'allowed':m.allowed}
    for (r,p),s in m.spaces.items():out[f'mean_{r}_{p}']=s.mean;out[f'basis_{r}_{p}']=s.basis;out[f'support_rank_{r}_{p}']=np.array([s.n,s.rank])
    for role,x in m.calibration.items():out[f'calibration_{role}']=x
    for state,x in getattr(m,'state_process_references',{}).items():out[f'process_reference_state_{state}']=x
    for name in ['request_calibration','route_calibration']:
        obj=getattr(m,name,None)
        if obj is not None:
            for (role,request),x in obj.references.items():out[f'{name}_{role}_{request}']=x
    if hasattr(m,'dwell'):
        out['dwell_reference']=m.dwell.reference
        for state,x in m.dwell.durations.items():out[f'dwell_state_{state}']=x
        for (a,b),x in m.dwell.context_durations.items():out[f'dwell_context_{a}_{b}']=x
        for (a,b),x in getattr(m.dwell,'log_parameters',{}).items():out[f'dwell_log_{a}_{b}']=np.array(x)
    return out

def setup():
    allcfg=[]
    for run in RUNS:
        for scene in SCENES:
            options=json.loads(Path(f'configs/experiment{TEMPLATES[scene]}.json').read_text());options.pop('appearance_phase_ranks',None)
            options.update(experiment=f'40_{run}_{scene}',scope=f'{scene}_shared_visual_representation_{run}',feature_source_experiment=f'40/{run}',representation_run=run,encoder='openai/clip-vit-base-patch32' if run=='A' else 'timm/MobileCLIP2-S2-OpenCLIP',encoder_frozen=run in ['A','B'])
            path=Path(f'configs/experiment40_{run}_{scene}.json')
            if path.exists():assert json.loads(path.read_text())==options
            else:write(path,options)
            link=Path(f'artifacts/experiment40_{run}_{scene}/features');link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to(f'../experiment40/{run}/features',target_is_directory=True)
            allcfg.append(path)
    return allcfg

def main():
    if (OUT/'normal_models_checkpoint.json').exists():raise RuntimeError('Normal stage already completed; verify existing checkpoint rather than overwrite')
    configs=setup();manifest=json.loads((OUT/'data_manifest.json').read_text());files=[*configs,Path(__file__),OUT/'representation_protocol.json']
    files += [discovery_path(cfg('A',scene)) for scene in SCENES]
    clip_record=Path('artifacts/model_paths.json');files.append(clip_record)
    clip=Path(json.loads(clip_record.read_text())['openai/clip-vit-base-patch32']['local_path'])
    files += sorted(clip.glob('*.json'))+sorted(clip.glob('*.safetensors'))+sorted(clip.glob('pytorch_model.bin'))
    for name,h in manifest['model']['sha256'].items():
        path=Path(manifest['model']['local_path'])/name;assert sha(path)==h;files.append(path)
    for run in RUNS:
        extraction=OUT/f'{run}_normal_extraction.json';record=json.loads(extraction.read_text());files.append(extraction)
        if run not in ['A','B']:
            training=OUT/f'{run}_training.json';info=json.loads(training.read_text());assert sha(info['checkpoint'])==info['checkpoint_sha256'];files.append(training)
        assert record['normal_only'] and record['views']==manifest['view_count']
        for row in record['sequences']:
            p=ART/run/'features'/row['scene']/f'training_{row["sequence"]}.npz';assert sha(p)==row['sha256'];files.append(p)
            options=json.loads(Path('configs/experiment40_representation.json').read_text());source=Path(options['source_features'][row['scene']])/p.name
            assert sha(source)==manifest['normal_source_feature_sha256'][str(source)]
            with np.load(source,allow_pickle=False) as f:original=dict(f)
            with np.load(p,allow_pickle=False) as f:current=dict(f)
            assert set(original)==set(current)
            for k,v in current.items():
                if k in ['global_features','crop_features']:
                    assert v.shape==original[k].shape and np.isfinite(v).all()
                    np.testing.assert_allclose(np.linalg.norm(v,axis=1),1,rtol=1e-5,atol=1e-6)
                else:np.testing.assert_array_equal(v,original[k])
    for p,h in json.loads((OUT/'representation_protocol.json').read_text())['file_sha256'].items():assert sha(p)==h,p
    record={'normal_only':True,'file_sha256':{str(p):sha(p) for p in files}}
    path=OUT/'pre_normal_models_protocol.json'
    if path.exists():assert json.loads(path.read_text())==record
    else:write(path,record)
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            s=args[0].decode() if isinstance(args[0],bytes) else args[0]
            if '/test_label/' in s or '/testing/frames/' in s or Path(s).name.startswith('testing_'):raise RuntimeError('Test access forbidden during normal model fitting')
    sys.addaudithook(hook);rows=[];full=[];dest=ART/'normal_models';dest.mkdir(parents=True,exist_ok=True);protected=[]
    for scene in SCENES:
        split=manifest['subsplits'][scene];control={}
        for run in RUNS:
            fit=[load(run,scene,s) for s in split['downstream_fit']];cal={s:load(run,scene,s) for s in split['normal_calibration']};base=factory(cfg(run,scene));base.fit(fit)
            for held in [None,*cal]:
                m=copy.deepcopy(base);used=[s for s in cal if s!=held];m.calibrate([cal[s] for s in used]);probe={s:cal[s] for s in cal if held is None or s==held};results={s:m.score(d) for s,d in probe.items()};keys=[k for k,v in next(iter(results.values())).items() if isinstance(v,np.ndarray)];saved={k:np.concatenate([r[k] for r in results.values()]) for k in keys};suffix=f'{run}_{scene}_{held or "full"}'
                references=arrays(m)
                for kind,values in [('model',references),('scores',saved)]:
                    path=dest/f'{suffix}_{kind}.npz';np.savez_compressed(path,**values);protected.append(path)
                process_keys=[k for k in keys if k not in ['visual','combined']];process_parameters={k:v for k,v in references.items() if k in ['transition','process_reference','allowed'] or k.startswith('process_reference_state_') or k.startswith('dwell_')}
                if run=='A':control[held]=({k:saved[k] for k in process_keys},process_parameters)
                else:
                    assert set(process_keys)==set(control[held][0])
                    for k in process_keys:np.testing.assert_array_equal(saved[k],control[held][0][k],err_msg=suffix+':'+k)
                    for k,v in process_parameters.items():np.testing.assert_array_equal(v,control[held][1][k],err_msg=suffix+':'+k)
                finite=all(np.isfinite(saved[k]).all() and np.all((saved[k]>=0)&(saved[k]<=1)) for k in ['visual','process','combined']);alarms=sum(int(hold_scores(probe[s]['indices'],results[s]['combined']>m.threshold,int(probe[s]['frame_count'])).sum()) for s in probe);count=sum(int(d['frame_count']) for d in probe.values())
                row={'run':run,'scene':scene,'held_out':held,'calibration_sequences':used,'q99':m.threshold,'finite_bounded':bool(finite),'eligible':bool(finite and m.threshold<1),'frames':count,'frame_alarms':alarms,'process_exact_to_A':True}
                if held is not None:assert held not in used;rows.append(row)
                else:
                    row['subspaces']=[{'role':r,'phase':p,'samples':v.n,'rank':v.rank} for (r,p),v in sorted(m.spaces.items())];row['dwell_context_support']=getattr(getattr(m,'dwell',None),'context_support',None);full.append(row)
            print(scene,run,'normal q99',full[-1]['q99'],'holdout FP',sum(r['frame_alarms'] for r in rows if r['scene']==scene and r['run']==run),flush=True)
    eligible=[f'{r["run"]}_{r["scene"]}' for r in full if r['eligible'] and all(v['eligible'] for v in rows if v['run']==r['run'] and v['scene']==r['scene'])]
    write(OUT/'normal_models_audit.json',{'normal_only':True,'full':full,'folds':rows,'eligible':eligible,'note':'Independent normal calibration video holdout; encoder representation-validation remains inside original downstream FIT. Process parameters/scores are exact within each scene across all visual arms. Actual PCA ranks may differ under the fixed variance/cap rule.'})
    protected += [OUT/'pre_normal_models_protocol.json',OUT/'normal_models_audit.json']
    write(OUT/'normal_models_checkpoint.json',{'file_sha256':{str(p):sha(p) for p in protected},'normal_only':True});print('Normal models frozen',len(full),len(rows),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
