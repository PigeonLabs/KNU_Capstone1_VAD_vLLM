"""Independent normal phase-fit reconstruction and fixed-case metadata review."""
import copy,json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from ipad_vad.transition_evidence import same_track_pair_transition_mask
from experiment37_asinh_phase import OUT,ART,SOURCE,GROUPS,SPLIT,root,models,load,sha,write,configure_common,common,normal_guard


def main():
    opened=normal_guard()
    for name in ['pre_normal_protocol.json','pre_calibration_checkpoint.json','pre_test_checkpoint.json','normal_verification_checkpoint.json']:common.verify_freeze(name)
    ms=models();fit=[load(SOURCE/f'training_{s}.npz') for s in SPLIT['fit']];x=np.concatenate([d['relation_descriptors'][d['relation_valid']] for d in fit]);time=np.concatenate([d['indices'][d['relation_valid']]/max(int(d['frame_count'])-1,1) for d in fit])
    med=np.median(x,axis=0);scale=np.maximum(np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0),.05);cl=KMeans(n_clusters=4,random_state=42,n_init=10).fit(np.arcsinh((x-med)/scale))
    order=np.argsort([np.median(time[cl.labels_==i]) for i in range(4)],kind='stable');saved=ms['asinh'];rare=np.flatnonzero(np.array(json.loads((OUT/'phase_fit.json').read_text())['normal_cluster_samples'])<10)
    for actual,expected in [(saved.location,med),(saved.scale,scale),(saved.centers,cl.cluster_centers_[order])]:np.testing.assert_array_equal(actual,expected)
    assert saved.area_upper==ms['control'].area_upper
    summary={};mapping=json.loads((OUT/'phase_fit.json').read_text())['diagnostic_asinh_to_control_mapping'];prefixes=0;cache={};sparse=[]
    sample_keys=['indices','global_features','phase_similarity','phases','relation_valid','relation_detection_indices','relation_descriptors']
    for part in ['fit','calibration']:
        mats={g:{name:np.zeros((4,4),int) for name in ['all','observed','same_pair']} for g in GROUPS};counts={g:np.zeros(4,int) for g in GROUPS};pervideo={g:{} for g in GROUPS};cont=np.zeros((4,4),int)
        for seq in SPLIT[part]:
            ds={g:load(root(g)/f'training_{seq}.npz') for g in GROUPS};cache[seq]=ds;a,b=ds.values();v=a['relation_valid'];np.add.at(cont,(a['phases'][v],b['phases'][v]),1)
            for g,d in ds.items():
                for k in a:
                    if k!='phases':np.testing.assert_array_equal(a[k],d[k])
                p=d['phases'];valid=d['relation_valid'];gate=same_track_pair_transition_mask(d);observed=valid[1:]&valid[:-1];counts[g]+=np.bincount(p[valid],minlength=4);pervideo[g][seq]=np.bincount(p[valid],minlength=4).tolist()
                for name,mask in [('all',np.ones(len(p)-1,bool)),('observed',observed),('same_pair',gate[1:])]:np.add.at(mats[g][name],(p[:-1][mask],p[1:][mask]),1)
                for end in [max(1,len(p)//2),len(p)-1]:
                    prefix={k:val[:end] if k in sample_keys else val if k=='frame_count' else val[d['object_frames']<end] for k,val in d.items()}
                    expected=ms[g].transform(prefix)
                    for val,key in zip(expected,['phases','relation_valid','relation_detection_indices','relation_descriptors']):np.testing.assert_array_equal(val,d[key][:end])
                    prefixes+=1
                if g=='asinh':
                    for i in np.flatnonzero(valid & np.isin(p,rare)):
                        selected=d['relation_detection_indices'][i];boxes=d['boxes'][selected];sparse.append({'partition':part,'sequence':seq,'sample':int(i),'source_frame':int(d['indices'][i]),'phase':int(p[i]),'selected_tracks':d['tracks'][selected].tolist(),'anchor_box':boxes[0].tolist(),'target_box':boxes[1].tolist(),'anchor_width_height':(boxes[0,2:]-boxes[0,:2]).tolist(),'descriptor':d['relation_descriptors'][i].tolist(),'phase_coordinate':np.arcsinh((d['relation_descriptors'][i]-saved.location)/saved.scale).tolist()})
        adjusted=sum(cont[int(old),int(new)] for new,old in mapping.items())
        summary[part]={'contingency_control_rows_asinh_columns':cont.tolist(),'raw_valid_agreement':float(np.trace(cont)/cont.sum()),'agreement_under_frozen_fit_permutation':float(adjusted/cont.sum()),'groups':{g:{'observed_phase_samples':counts[g].tolist(),'transition_counts':{k:v.tolist() for k,v in mats[g].items()},'per_video_phase_counts':pervideo[g],'distinct_videos_per_phase':np.sum(np.array(list(pervideo[g].values()))>0,axis=0).tolist(),'largest_video_share_per_phase':np.divide(np.max(list(pervideo[g].values()),axis=0),counts[g],out=np.zeros(4,float),where=counts[g]>0).tolist()} for g in GROUPS}}
    cont=np.array(summary['fit']['contingency_control_rows_asinh_columns']);a,b=linear_sum_assignment(-cont);assert {str(new):int(old) for old,new in zip(a,b)}==mapping
    source=json.loads(Path('results/experiment36/fixed_case_phase_review.json').read_text());prior=json.loads(Path('results/experiment36/pre_evaluation_review_checkpoint.json').read_text());assert sha('results/experiment36/fixed_case_phase_review.json')==prior['file_sha256']['results/experiment36/fixed_case_phase_review.json']
    cases=[];changed=validchanged=0
    natural=json.loads((OUT/'fit_rank_control.json').read_text())['natural_banks']
    for case in source['cases']:
        seq=case['case']['sequence'];frames=[]
        for f in case['frames']:
            i=f['sample'];a,b=cache[seq].values();valid=bool(a['relation_valid'][i]);np.testing.assert_array_equal(a['relation_detection_indices'][i],f['selected_indices']);assert a['phases'][i]==f['phases']['refit'];assert a['indices'][i]==f['source_frame'];pc=int(a['phases'][i]);pn=int(b['phases'][i]);changed+=pc!=pn;validchanged+=valid and pc!=pn
            cell={'sample':i,'source_frame':f['source_frame'],'source_path_reference':f['source_path_reference'],'valid':valid,'selection_unchanged':True,'selected_indices':a['relation_detection_indices'][i].tolist(),'phases':{'control':pc,'asinh':pn},'asinh_phase_mapped_for_diagnostic_only':mapping[str(pn)]}
            cell['groups']={}
            for g,d in cache[seq].items():
                phase=int(d['phases'][i]);m=ms[g];coordinate=(d['relation_descriptors'][i]-m.location)/m.scale;coordinate=np.arcsinh(coordinate) if g=='asinh' else coordinate;dist=np.sum((coordinate-m.centers)**2,axis=1).tolist() if valid else None
                cell['groups'][g]={'squared_distances':dist,'appearance_phase_bank_samples':{str(role):natural[g].get(f'{role}:{phase}',{}).get('samples',0) for role in [-1,0,1,2]}}
            frames.append(cell)
        cases.append({'case':case['case'],'frames':frames,'semantic_ground_truth':None,'action_boundary_ground_truth':None})
    write(OUT/'normal_phase_audit.json',{'partitions':summary,'fit_model_exactly_reconstructed':True,'prefix_checks':prefixes,'raw_feature_selection_valid_descriptor_unchanged':True,'sparse_cluster_observations':sparse,'source_case_metadata_sha256':sha('results/experiment36/fixed_case_phase_review.json'),'normal_data_opened':sorted(opened),'test_accessed':False})
    write(OUT/'fixed_case_phase_review.json',{'normal_only':True,'case_source':'results/experiment36/fixed_case_phase_review.json','case_source_sha256':sha('results/experiment36/fixed_case_phase_review.json'),'cases':cases,'case_count':len(cases),'frames':sum(len(c['frames']) for c in cases),'raw_id_changed_frames':changed,'valid_raw_id_changed_frames':validchanged,'new_images_reviewed':False,'note':'Same 24 previously inspected cases, unchanged detections and selection. This is phase/distance/support metadata comparison, not a new visual or semantic accuracy review. Cluster IDs differ and squared distances use linear/asinh coordinates.'})
    common.freeze('pre_evaluation_review_checkpoint.json',[OUT/'normal_verification_checkpoint.json',OUT/'normal_phase_audit.json',OUT/'fixed_case_phase_review.json',Path(__file__)])
    print(json.dumps({'phase_summary':summary,'case_frames_changed_ids':changed,'prefix_checks':prefixes,'sparse_observations':sparse},indent=2),flush=True)

if __name__=='__main__':
    configure_common()
    with threadpool_limits(limits=4):main()
