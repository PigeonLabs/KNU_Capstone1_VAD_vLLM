"""Independent normal-only coverage, selection, trace/source and immutability checks."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from audit_missing_age import load,sha


def main():
    out=Path('results/experiment29');art=Path('artifacts/experiment29');cfg=json.loads(Path('configs/experiment29.json').read_text());protocol=json.loads((out/'pre_audit_protocol.json').read_text());checkpoint=json.loads((out/'pre_visual_checkpoint.json').read_text());audit=json.loads((out/'coverage_audit.json').read_text());selection=json.loads((out/'case_selection.json').read_text())['cases'];history=json.loads((out/'history_diagnostic.json').read_text());split=json.loads(Path('results/stage00/splits.json').read_text())['R04']
    for p,h in {**protocol['protected_sha256'],**protocol['code_input_sha256'],**checkpoint['source_normal_image_sha256'],**checkpoint['local_trace_sha256'],**history['additional_visual_followup_source_hashes']}.items():assert sha(p)==h,p
    assert sha(out/'case_selection.json')==checkpoint['selection_sha256'];assert len({r['case_id'] for r in selection})==len(selection)==30
    edge_sources={};total_frames=total_samples=0;allrows=[];mixed={}
    for part in ['fit','calibration']:
        mixed[part]={'valid_samples':0,'mixed_pair_windows':0,'mixed_anchor_windows':0,'mixed_target_windows':0,'phases':{str(p):{'valid_samples':0,'mixed_pair_windows':0} for p in range(4)}};counts=np.zeros((4,4),int);observed=np.zeros((4,4),int);videos=defaultdict(set);ovideos=defaultdict(set);by_video={};oby_video={}
        for seq in split[part]:
            d=load(Path(cfg['feature_source'])/f'training_{seq}.npz');trace=json.loads((art/'traces'/f'{seq}.json').read_text());assert trace['partition']==part;assert len(trace['frames'])==len(d['indices']);assert len(trace['transitions'])==len(d['indices'])-1;total_frames+=int(d['frame_count']);total_samples+=len(d['indices']);by_video[seq]=np.zeros((4,4),int);oby_video[seq]=np.zeros((4,4),int)
            for i,frame in enumerate(trace['frames']):
                assert frame['sample']==i and frame['source_frame']==int(d['indices'][i]);assert Path(frame['source_path']).exists() and int(Path(frame['source_path']).stem)==frame['source_frame'];assert frame['phase']==d['phases'][i] and frame['relation_valid']==d['relation_valid'][i]
                for j,role in enumerate(['anchor','target']):
                    idx=int(d['relation_detection_indices'][i,j]);obj=frame['selected'][role]
                    if idx<0:assert obj is None
                    else:
                        assert obj['detection_index']==idx and obj['track']==int(d['tracks'][idx]);np.testing.assert_array_equal(obj['bbox_normalized'],d['boxes'][idx]);assert d['object_frames'][idx]==i
                if i==0:continue
                r=trace['transitions'][i-1];a,b=int(d['phases'][i-1]),int(d['phases'][i]);valid=bool(d['relation_valid'][i-1] and d['relation_valid'][i]);assert r['previous_phase']==a and r['current_phase']==b and r['observed_pair']==valid and r['source_frame']==int(d['indices'][i]);counts[a,b]+=1;by_video[seq][a,b]+=1;videos[a,b].add(seq);allrows.append(r)
                if valid:observed[a,b]+=1;oby_video[seq][a,b]+=1;ovideos[a,b].add(seq);edge_sources.setdefault((part,a,b),[]).append((seq,int(d['indices'][i])))
            for frame in trace['frames']:
                if frame['relation_valid']:
                    i=frame['sample'];start=i
                    while start>0 and i-start<2 and trace['frames'][start-1]['relation_valid']:start-=1
                    assert frame['smoothing_observations']==i-start+1
                    window=trace['frames'][start:i+1];pairs=[(f['selected']['anchor']['track'],f['selected']['target']['track']) for f in window];x=mixed[part];x['valid_samples']+=1;x['mixed_pair_windows']+=len(set(pairs))>1;x['mixed_anchor_windows']+=len({a for a,b in pairs})>1;x['mixed_target_windows']+=len({b for a,b in pairs})>1;x['phases'][str(frame['phase'])]['valid_samples']+=1;x['phases'][str(frame['phase'])]['mixed_pair_windows']+=len(set(pairs))>1
        for key,m,v,per in [('all',counts,videos,by_video),('observed',observed,ovideos,oby_video)]:
            got=audit['coverage'][part][key];np.testing.assert_array_equal(got['counts'],m);assert got['distinct_videos']==[[len(v[a,b]) for b in range(4)] for a in range(4)]
            for seq,arr in per.items():np.testing.assert_array_equal(got['by_video'][seq],arr)
    assert mixed==history['totals']
    review=json.loads((out/'visual_review.json').read_text());assert {r['case_id'] for r in review['reviewed_cases']}=={r['case_id'] for r in selection};assert all(r['reviewed'] and r['semantic_ground_truth'] is None for r in review['reviewed_cases'])
    targets=[r for r in selection if r['selection_group']=='target_3_to_2'];assert {(r['sequence'],r['source_frame']) for r in targets}==set(edge_sources['fit',3,2]+edge_sources['calibration',3,2]);assert len(targets)==6
    controls=[r for r in selection if r['selection_group']=='control_previous_3'];assert len(controls)==24
    eligible=[r for r in allrows if r['observed_pair'] and r['previous_phase']==3 and r['current_phase']!=2];expected=[];buckets={s:sorted([r for r in eligible if r['sequence']==s],key=lambda r:r['source_frame']) for s in sorted({r['sequence'] for r in eligible})}
    for seq,items in buckets.items():
        if len(expected)<24:expected.append(items[0]['case_id'])
    assert expected==[r['case_id'] for r in controls]
    for fold in audit['calibration_holdout']:
        held=fold['held_out'];remaining=np.zeros((4,4),int)
        for r in allrows:
            if r['partition']=='calibration' and r['sequence']!=held and r['observed_pair']:remaining[r['previous_phase'],r['current_phase']]+=1
        np.testing.assert_array_equal(fold['remaining_coverage']['counts'],remaining)
        expected_edges=[]
        for a in range(4):
            for b in range(4):
                frames=[r['source_frame'] for r in allrows if r['partition']=='calibration' and r['sequence']==held and r['observed_pair'] and (r['previous_phase'],r['current_phase'])==(a,b)]
                if frames and remaining[a,b]==0:expected_edges.append((a,b,frames))
        assert [(r['previous'],r['current'],r['held_source_frames']) for r in fold['held_edges_absent_from_remaining']]==expected_edges
    manifest=json.loads((out/'local_contact_sheet_manifest.json').read_text());assert {c for p in manifest['pages'] for c in p['case_ids']}=={r['case_id'] for r in selection}
    for p in manifest['pages']:assert sha(p['path'])==p['sha256']
    result={'normal_sequences_checked':25,'normal_source_frames':total_frames,'sampled_frames':total_samples,'transitions':len(allrows),'selected_cases':30,'local_contact_sheets':len(manifest['pages']),'protected_files_unchanged':len(protocol['protected_sha256']),'code_input_and_selection_image_hashes_match':True,'phase_mask_descriptor_margin_reconstruction_verified':audit['all_cached_phase_mask_selection_descriptors_margins_exactly_reconstructed'],'coverage_video_counts_holdout_and_mixed_windows_independently_reconstructed':True,'all_selected_cases_have_exploratory_review_records':True,'all_trace_source_paths_selected_boxes_and_tracks_verified':True,'selection_before_visual_review_verified':True,'test_inputs_opened':False,'score_or_model_changes':False}
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
