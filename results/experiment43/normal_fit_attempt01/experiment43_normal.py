"""Replay all visual arms with frozen association metadata and process models."""
import copy,json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.data import hold_scores
from ipad_vad.learned_detector import sha,write
from experiment40_normal import SCENES,RUNS,arrays
from experiment41_normal import factory,cfg as original_cfg
from ipad_vad.experiment import load_process
from experiment41_phases import load_npz
OUT=Path('results/experiment43');ART=Path('artifacts/experiment43');BRANCHES=['teacher','linear','adapter']

def cfg(run,scene,branch='teacher'):
    options=original_cfg(run,scene);options.update(experiment=f'43_{branch}_{run}_{scene}',phase_model_branch=branch,process_discovery=f'results/experiment43/process_{scene}.json');return options

def setup():
    for scene in SCENES:
        process=load_process(original_cfg('A',scene))
        if scene=='R02':
            process['phases']=[{'id':f'latent_{i}','description':f'Normal platform-scissor relation cluster {i}; not a verified action label.'} for i in range(4)];process['normal_order']=[f'latent_{i}' for i in range(4)]
        process['phase_target_note']='Experiment43: normal-train relation/spatial centers, class definitions not action ground truth; same self/next/cycle template.';write(OUT/f'process_{scene}.json',process)
    for branch in BRANCHES:
        for scene in SCENES:
            for run in RUNS:write(f'configs/experiment43_{branch}_{run}_{scene}.json',cfg(run,scene,branch))

def load(branch,run,scene,seq,split='training'):
    d=load_npz(Path('artifacts/experiment41')/run/'features'/scene/f'{split}_{seq}.npz')
    if branch!='iou':d.update(load_npz(ART/branch/'observations'/scene/f'{split}_{seq}.npz'))
    d['sequence_id']=f'{scene}/{split}_{seq}';return d

def verify(record):
    for p,h in json.loads(Path(record).read_text())['file_sha256'].items():assert sha(p)==h,p

def main():
    if (OUT/'normal_models_checkpoint.json').exists():raise RuntimeError('Normal already frozen')
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            s=str(args[0])
            if '/test_label/' in s or '/testing/' in s or Path(s).name.startswith('testing_'):raise RuntimeError('Normal only')
    sys.addaudithook(guard);setup();verify(OUT/'training_protocol.json');verify(OUT/'training_observations.json');manifest=json.loads(Path('results/experiment40/data_manifest.json').read_text());old_audit=json.loads(Path('results/experiment41/normal_models_audit.json').read_text());files=[Path(__file__),OUT/'training_observations.json',OUT/'training.json',OUT/'target_review.json',Path('scripts/experiment40_normal.py'),Path('scripts/experiment41_normal.py'),Path('scripts/evaluate_phase43.py')]+sorted(Path('src/ipad_vad').glob('*.py'))+list(Path('configs').glob('experiment43_*.json'))+[OUT/f'process_{s}.json' for s in SCENES]+[Path('scripts/phase43_common.py'),Path('scripts/extract_phase43.py')];write(OUT/'pre_normal_models_protocol.json',{'normal_only':True,'file_sha256':{str(p):sha(p) for p in files}});full=[];folds=[];protected=[];control_checks=0;all_exact=True
    for scene in SCENES:
        sp=manifest['subsplits'][scene]
        for run in RUNS:
            for branch in BRANCHES:
                fit=[load(branch,run,scene,s) for s in sp['downstream_fit']];cal={s:load(branch,run,scene,s) for s in sp['normal_calibration']};base=factory(cfg(run,scene,branch));base.fit(fit)
                for held in [None,*cal]:
                    m=copy.deepcopy(base);used=[s for s in cal if s!=held];m.calibrate([cal[s] for s in used]);probe={s:d for s,d in cal.items() if held is None or s==held};result={s:m.score(d) for s,d in probe.items()};keys=[k for k,v in next(iter(result.values())).items() if isinstance(v,np.ndarray)];scores={k:np.concatenate([r[k] for r in result.values()]) for k in keys};params=arrays(m);stem=f'{run}_{scene}_{held or "full"}';same=True
                    for kind,values in [('model',params),('scores',scores)]:
                        old=load_npz(Path('artifacts/experiment41/normal_models')/f'{stem}_{kind}.npz');equal=set(old)==set(values) and all(np.array_equal(old[k],v) for k,v in values.items());same&=equal
                        if branch=='iou':assert equal,(stem,kind);control_checks+=1
                        else:
                            p=ART/branch/'normal_models'/f'{stem}_{kind}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**values);protected.append(p)
                    all_exact&=same;alarms=sum(int(hold_scores(probe[s]['indices'],result[s]['combined']>m.threshold,int(probe[s]['frame_count'])).sum()) for s in probe);frames=sum(int(d['frame_count']) for d in probe.values());finite=all(np.isfinite(scores[k]).all() and ((scores[k]>=0)&(scores[k]<=1)).all() for k in ['visual','process','combined']);row={'branch':branch,'run':run,'scene':scene,'held_out':held,'q99':m.threshold,'finite_bounded':bool(finite),'eligible':bool(finite and m.threshold<1),'frames':frames,'frame_alarms':alarms,'exact_model_and_scores_to_41':bool(same)}
                    if held:folds.append(row)
                    else:row.update(dwell_available=getattr(getattr(m,'dwell',None),'available',None),dwell_unavailable_reason=getattr(getattr(m,'dwell',None),'unavailable_reason',None));full.append(row)
                print(scene,run,branch,'q99',m.threshold,flush=True)
    write(OUT/'normal_models_audit.json',{'normal_only':True,'full':full,'folds':folds,'all_new_models_and_scores_exact_to_41':bool(all_exact),'control_model_score_array_files_exact':control_checks,'holdout_scope':'Calibration video omitted from CDF/q99; phase-head optimization/checkpoint selection exclude all22 calibration videos. Legacy frozen observation gates/scalers may include representation-validation videos inside original FIT; weak-target validation is not independent action-GT validation.'});protected +=[OUT/'normal_models_audit.json',OUT/'pre_normal_models_protocol.json'];write(OUT/'normal_models_checkpoint.json',{'normal_only':True,'file_sha256':{str(p):sha(p) for p in protected}});print('NORMAL FROZEN',len(full),len(folds),'exact41',all_exact,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
