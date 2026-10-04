"""Staged comparison of causal confirmation versus immediate semantic selection."""
import argparse,copy,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
import numpy.testing
from threadpoolctl import threadpool_limits
from ipad_vad.semantic_priority_anchor import SemanticPriorityAnchorPhase
from ipad_vad.confirmed_anchor import ConfirmedAnchorPhase
from ipad_vad.scoring import Baseline
from ipad_vad.experiment import load_process
from ipad_vad.data import hold_scores
from ipad_vad.dwell_episodes import extract_dwell_episodes
from audit_transition_provenance import restore_model
from experiment32_duration_evidence import protected_paths
import experiment30_track_reset as common
from experiment30_track_reset import write,load,sha

OUT=Path('results/experiment35');ART=Path('artifacts/experiment35');SOURCE=Path('artifacts/experiment34/semantic/features/R04');GROUPS=['control','confirmed'];GATES=['hold','pool','age'];VARIANTS=[f'35_{g}_{a}' for g in GROUPS for a in GATES];SPLIT=common.SPLIT

def root(g):return ART/g/'features/R04'
def protected_inputs():
    return protected_paths()+[*[SOURCE/f'training_{s}.npz' for s in SPLIT['fit']+SPLIT['calibration']],*[Path(f'artifacts/experiment34_semantic_{gate}/{name}.npz') for gate in GATES for name in ['normal_model','normal_calibration_scores']],*sorted(Path('artifacts/experiment34/full_normal').glob('*.npz')),*sorted(Path('artifacts/experiment34/normal_holdout').glob('*.npz'))]

def cfg(e):return json.loads(Path(f'configs/experiment{e}.json').read_text())
def configure_common():
    common.OUT=OUT;common.ART=ART;common.GROUPS=GROUPS;common.GATES=GATES;common.VARIANTS=VARIANTS;common.root=root;common.cfg=cfg

def normal_guard():
    allowed={str(p.resolve()) for p in protected_inputs()+[Path('artifacts/experiment18/normal_relation_model.npz'),Path('artifacts/experiment17/anchor_text_features.npz')]};opened=set()
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]);s=str(p.resolve())
        if '/test_label/' in s or '/predictions/' in s or p.name.startswith('testing_') or ('/testing/' in s and p.suffix not in ['.py','.pyc']):raise RuntimeError('Test access forbidden before normal checkpoint: '+s)
        if p.suffix.lower() in ['.npz','.npy','.jpg','.png','.mp4']:
            if s not in allowed and not p.resolve().is_relative_to(ART.resolve()):raise RuntimeError('Outside normal numeric allowlist: '+s)
            opened.add(s)
    sys.addaudithook(hook);return opened

def models():
    saved=restore_model({'relation_config':'configs/experiment18.json','relation_model':'artifacts/experiment18/normal_relation_model.npz'})
    text=load('artifacts/experiment17/anchor_text_features.npz')['text_features'];kwargs=dict(window=saved.window,margin_threshold=saved.margin_threshold,**cfg('34_semantic_hold')['relational_phase'])
    old=SemanticPriorityAnchorPhase(text,**kwargs);new=ConfirmedAnchorPhase(text,confirmation_samples=2,**kwargs)
    for m in [old,new]:
        for key in ['location','scale','centers','area_upper']:setattr(m,key,copy.deepcopy(getattr(saved,key)))
    return {'control':old,'confirmed':new}

def independently_check(d,m,group,p,v,c,x):
    margins=m.margins(d);prior=[None,None];history=[];last=None;phase=0;expected=np.full_like(c,-1);pending_track=None;pending_count=0;pending_step=None;decisions=[]
    for step in range(len(d['indices'])):
        for col,role in enumerate([m.anchor_role,m.target_role]):
            ids=[]
            for i in np.flatnonzero((d['object_frames']==step)&(d['roles']==role)):
                area=float(np.maximum(d['boxes'][i,2:]-d['boxes'][i,:2],0).prod())
                if area>0 and area<=m.area_upper[role]+1e-8 and (col==1 or margins[i]>m.margin_threshold):ids.append(int(i))
            if ids:
                if col==0:
                    best=max(ids,key=lambda i:(float(margins[i]),int(d['tracks'][i])==prior[col],float(d['confidence'][i]),-i));index=best
                    same=[i for i in ids if int(d['tracks'][i])==prior[col]];reason='forced' if not same else 'retained' if int(d['tracks'][best])==prior[col] else 'immediate_challenger';count=0
                    if group=='confirmed':
                        if not same or int(d['tracks'][best])==prior[col]:pending_track=None;pending_count=0;pending_step=None
                        else:
                            winner=int(d['tracks'][best]);count=pending_count+1 if pending_track==winner and pending_step==step-1 else 1
                            if count>=2:reason='confirmed_challenger';pending_track=None;pending_count=0;pending_step=None
                            else:index=max(same,key=lambda i:(float(d['confidence'][i]),-i));reason='pending_challenger';pending_track=winner;pending_count=count;pending_step=step
                    decisions.append({'sample':step,'source_frame':int(d['indices'][step]),'reason':reason,'best_track':int(d['tracks'][best]),'selected_track':int(d['tracks'][index]),'previous_track':prior[col],'confirmation_count':count,'best_margin':float(margins[best]),'selected_margin':float(margins[index])})
                else:index=max(ids,key=lambda i:(int(d['tracks'][i])==prior[col],float(d['confidence'][i]),-i))
                expected[step,col]=index;prior[col]=int(d['tracks'][index])
            elif col==0:
                pending_track=None;pending_count=0;pending_step=None;decisions.append({'sample':step,'source_frame':int(d['indices'][step]),'reason':'missing','best_track':None,'selected_track':None,'previous_track':prior[col],'confirmation_count':0,'best_margin':None,'selected_margin':None})
        assert v[step]==bool(np.all(expected[step]>=0))
        if v[step]:
            pair=tuple(d['tracks'][expected[step]])
            if pair!=last:history=[]
            last=pair;a,b=d['boxes'][expected[step]];sa=a[2:]-a[:2];sb=b[2:]-b[:2];ca=(a[:2]+a[2:])/2;cb=(b[:2]+b[2:])/2;aa=sa.prod();ab=sb.prod();inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();raw=np.r_[(cb-ca)/np.maximum(sa,1e-8),np.log(ab/aa),inter/max(aa+ab-inter,1e-8),ca];history=(history+[raw])[-m.smoothing:]
            np.testing.assert_array_equal(x[step],np.mean(history,axis=0));phase=int(np.argmin(np.sum(((x[step]-m.location)/m.scale-m.centers)**2,axis=1)))
        else:history=[];last=None;assert not x[step].any()
        assert p[step]==phase
    np.testing.assert_array_equal(c,expected);return decisions

def transform(part,sequences):
    ms=models();rows=[];eps=[]
    for seq in sequences:
        d=load(SOURCE/f'{part}_{seq}.npz');results={};details={};partition=('fit' if seq in SPLIT['fit'] else 'calibration') if part=='training' else 'test'
        for g,m in ms.items():
            p,v,c,x=m.transform(d);decisions=independently_check(d,m,g,p,v,c,x);write(ART/'selection_traces'/g/f'{part}_{seq}.json',decisions)
            np.testing.assert_array_equal(m.margins(d),d['anchor_gate_margin']);np.testing.assert_array_equal(m.raw_margins(d),d['anchor_margin'])
            if g=='control':
                for key,val in [('phases',p),('relation_valid',v),('relation_detection_indices',c),('relation_descriptors',x)]:np.testing.assert_array_equal(d[key],val)
            derived=dict(d,phases=p,relation_valid=v,relation_detection_indices=c,relation_descriptors=x)
            for k in d:
                if k not in ['phases','relation_detection_indices','relation_descriptors']:np.testing.assert_array_equal(d[k],derived[k])
            np.testing.assert_array_equal(c[:,1],d['relation_detection_indices'][:,1]);np.testing.assert_array_equal(v,d['relation_valid'])
            dest=root(g)/f'{part}_{seq}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                saved=load(dest)
                for k in derived:np.testing.assert_array_equal(derived[k],saved[k])
            else:np.savez_compressed(dest,**derived)
            episode=extract_dwell_episodes(derived)['episodes'];eps.extend(dict(e,group=g,partition=partition,sequence=seq) for e in episode)
            pair_changes=Counter()
            for i in range(1,len(v)):
                if v[i-1] and v[i]:
                    change=d['tracks'][c[i]]!=d['tracks'][c[i-1]]
                    if change.any():pair_changes['both' if change.all() else 'anchor' if change[0] else 'target']+=1
            validanchor=c[:,0]>=0
            details[g]={'valid_samples':int(v.sum()),'pair_changes':dict(pair_changes),'phase_counts':np.bincount(p,minlength=4).tolist(),'observed_phase_counts':np.bincount(p[v],minlength=4).tolist(),'episode_status':dict(Counter(e['status'] for e in episode)),'selection_decisions':dict(Counter(r['reason'] for r in decisions)),'selected_anchor_margin_mean':float(np.mean(d['anchor_gate_margin'][c[validanchor,0]])) if validanchor.any() else None}
            results[g]=derived
        a,b=results['control'],results['confirmed'];change=a['relation_detection_indices'][:,0]!=b['relation_detection_indices'][:,0];phase=a['phases']!=b['phases']
        rows.append({'sequence':seq,'partition':partition,'samples':len(v),'frames':int(d['frame_count']),'anchor_selection_changed_samples':int(change.sum()),'phase_changed_samples':int(phase.sum()),'phase_changed_frames':int(hold_scores(d['indices'],phase,int(d['frame_count'])).sum()),'descriptor_changed_samples':int(np.any(a['relation_descriptors']!=b['relation_descriptors'],axis=1).sum()),'groups':details})
    return rows,eps

def prepare():
    opened=normal_guard();files=[Path('docs/EXPERIMENT35_PLAN.md'),Path('scripts/experiment35_confirmed_anchor.py'),Path('tests/test_confirmed_anchor.py'),*sorted(Path('src/ipad_vad').glob('*.py')),*[Path('scripts')/n for n in ['experiment30_track_reset.py','evaluate_baseline.py','audit_transition_provenance.py','audit_bank_dispatch.py','evaluate_route_holdout.py','audit_missing_age.py']],*[Path(f'configs/experiment34_semantic_{g}.json') for g in GATES],Path('configs/experiment18.json'),Path('results/stage00/splits.json'),Path('results/experiment18/process_discovery.json'),Path('artifacts/experiment18/normal_relation_model.npz'),Path('artifacts/experiment17/anchor_text_features.npz'),*protected_inputs()]
    common.freeze('pre_normal_protocol.json',files)
    rows,episodes=transform('training',SPLIT['fit']+SPLIT['calibration']);write(OUT/'normal_transform.json',{'normal_only':True,'rows':rows,'valid_and_target_selection_unchanged':True,'independent_scalar_selection_descriptor_phase_checked':True});write(OUT/'normal_episodes.json',{'episodes':episodes})
    natural={};fitted={}
    for g in GROUPS:
        options=cfg('34_semantic_hold');options.pop('appearance_phase_ranks');m=Baseline(options,load_process(options));m.fit([dict(load(root(g)/f'training_{s}.npz'),sequence_id=f'R04/training_{s}') for s in SPLIT['fit']]);fitted[g]=m
        natural[g]={f'{r}:{p}':{'samples':v.n,'rank':v.rank} for (r,p),v in m.spaces.items()}
    limits=cfg('34_semantic_hold')['appearance_phase_ranks'];maps={g:{} for g in GROUPS}
    for g in GROUPS:
        other='confirmed' if g=='control' else 'control'
        for key,value in natural[g].items():
            if key.endswith(':-1'):continue
            peer=natural[other].get(key);maps[g][key]=min(value['rank'],peer['rank'] if peer else value['rank'],limits.get(key,32))
    for key,space in fitted['control'].spaces.items():
        if key[1]==-1:
            for attr in ['mean','basis']:np.testing.assert_array_equal(getattr(space,attr),getattr(fitted['confirmed'].spaces[key],attr))
    write(OUT/'fit_rank_control.json',{'normal_fit_only':True,'natural_banks':natural,'rank_maps':maps,'previous_rank_limits':limits,'control_rank_map_matches_34':maps['control']==limits,'pooled_banks_unchanged':True,'shared_phase_banks':sorted(set(maps['control'])&set(maps['confirmed'])),'only_control':sorted(set(maps['control'])-set(maps['confirmed'])),'only_confirmed':sorted(set(maps['confirmed'])-set(maps['control']))})
    for g in GROUPS:
        for gate in GATES:
            e=f'35_{g}_{gate}';options=cfg(f'34_semantic_{gate}');options.update(experiment=e,scope=f'R04_anchor_selection_{g}_{gate}',feature_source_experiment=f'35/{g}',appearance_phase_ranks=maps[g],anchor_selection='confirmed_semantic_margin' if g=='confirmed' else 'semantic_margin_then_track_confidence_index',anchor_confirmation_samples=2 if g=='confirmed' else 1);write(f'configs/experiment{e}.json',options)
            link=Path(f'artifacts/experiment{e}/features');link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to(f'../experiment35/{g}/features',target_is_directory=True)
    common.freeze('pre_calibration_checkpoint.json',[OUT/'pre_normal_protocol.json',OUT/'normal_transform.json',OUT/'normal_episodes.json',OUT/'fit_rank_control.json',*[Path(f'configs/experiment{e}.json') for e in VARIANTS],*[p for g in GROUPS for p in sorted(root(g).glob('training_*.npz'))]])
    write(OUT/'prepare_access.json',{'opened_normal_data':sorted(opened),'test_data_opened':False});print(json.dumps({'rank_maps':maps,'normal_sequences':len(rows)},indent=2),flush=True)

def normal():
    opened=normal_guard();common.normal()
    ranks=json.loads((OUT/'fit_rank_control.json').read_text());checks={}
    if ranks['control_rank_map_matches_34']:
        for gate in GATES:
            e=f'35_control_{gate}';saved=load(ART/'full_normal'/f'{e}_model.npz');old=load(f'artifacts/experiment34_semantic_{gate}/normal_model.npz')
            for k,v in saved.items():np.testing.assert_array_equal(v,old[k],err_msg=k)
            scores=load(ART/'full_normal'/f'{e}_scores.npz');previous=load(f'artifacts/experiment34_semantic_{gate}/normal_calibration_scores.npz')
            for k,v in scores.items():np.testing.assert_array_equal(v,previous[k],err_msg=k)
            checks[e]='model and calibration scores exactly reproduce experiment34_semantic'
    else:checks['control']='rank changed; compare refitted matched-rank control, not identical to experiment34_semantic'
    write(OUT/'control_reproduction.json',checks);write(OUT/'normal_access.json',{'opened_normal_data':sorted(opened),'test_data_opened':False})
    common.freeze('normal_verification_checkpoint.json',[OUT/'pre_test_checkpoint.json',OUT/'control_reproduction.json',OUT/'normal_access.json'])

def test_prepare():
    for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json','pre_test_checkpoint.json','normal_verification_checkpoint.json','pre_evaluation_review_checkpoint.json']:common.verify_freeze(name)
    audit=json.loads((OUT/'normal_audit.json').read_text());assert audit['eligible']
    paths=sorted(SOURCE.glob('testing_*.npz'));common.freeze('test_input_checkpoint.json',paths)
    rows,episodes=transform('testing',[p.stem.split('_')[1] for p in paths]);write(OUT/'test_transform.json',{'rows':rows,'no_labels_used_for_transform':True,'eligible_variants':audit['eligible']});write(OUT/'test_episode_diagnostic.json',{'episodes':episodes,'not_fit_or_calibration_data':True})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','normal','test_prepare']);args=parser.parse_args();configure_common()
    with threadpool_limits(limits=4):globals()[args.stage]()
