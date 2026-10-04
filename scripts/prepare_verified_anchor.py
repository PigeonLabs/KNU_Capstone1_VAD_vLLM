"""Audit normal support first, then transform frozen features with a role gate."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.verified_anchor import VerifiedAnchorPhase
from ipad_vad.context_dwell import complete_context_runs


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    p=argparse.ArgumentParser();p.add_argument('--all-sequences',action='store_true');args=p.parse_args()
    cfg=json.loads(Path('configs/experiment17.json').read_text());scene=cfg['scene'];split=json.loads(Path('results/stage00/splits.json').read_text())[scene];opt=cfg['anchor_verification']
    encoding=json.loads(Path('results/experiment17/text_encoding.json').read_text());assert hashlib.sha256(Path(opt['text_features']).read_bytes()).hexdigest()==encoding['text_features_sha256']
    source=Path(f'artifacts/experiment16/features/{scene}');out=Path('results/experiment17');target=Path(f'artifacts/experiment17/features/{scene}')
    model=VerifiedAnchorPhase(load(opt['text_features'])['text_features'],margin_threshold=opt['margin_threshold'],**cfg['relational_phase'])
    data={s:load(source/f'training_{s}.npz') for s in split['fit']};rows=[];qual=[]
    for seq,d in data.items():
        margins=model.margins(d);anchor=d['roles']==model.anchor_role;keep=anchor&(margins>model.margin_threshold)
        rows.append({'sequence':seq,'sampled_frames':len(d['indices']),'anchor_boxes_before':int(anchor.sum()),'anchor_boxes_after':int(keep.sum()),'anchor_observed_samples_before':len(np.unique(d['object_frames'][anchor])),'anchor_observed_samples_after':len(np.unique(d['object_frames'][keep])),'anchor_margin_quantiles':np.quantile(margins[anchor],[0,.25,.5,.75,1]).tolist() if anchor.any() else None})
        if seq in ('01','03'):
            for step in [0,len(d['indices'])//2,len(d['indices'])-1]:
                ids=np.flatnonzero(anchor&(d['object_frames']==step));qual.append({'sequence':seq,'source_frame':int(d['indices'][step]),'old_selected_anchor_detection':int(d['relation_detection_indices'][step,0]),'candidates':[{'detection_index':int(i),'track':int(d['tracks'][i]),'box':d['boxes'][i].tolist(),'margin':float(margins[i]),'accepted':bool(keep[i])} for i in ids]})
    failure=None;transformed={};contexts={}
    try:
        with threadpool_limits(limits=4):model.fit(list(data.values()))
        for seq,d in data.items():
            phase,valid,chosen,x=model.transform(d);derived=dict(d,phases=phase,relation_valid=valid,relation_detection_indices=chosen,relation_descriptors=x,anchor_margin=model.margins(d));transformed[seq]=derived
            row=next(r for r in rows if r['sequence']==seq);row.update(valid_relations_before=int(d['relation_valid'].sum()),valid_relations_after=int(valid.sum()),phase_counts=np.bincount(phase,minlength=model.k).tolist())
            for (a,b),duration in complete_context_runs(derived):contexts.setdefault(f'{a}->{b}',[]).append({'sequence':seq,'duration_frames':duration})
    except ValueError as e:failure=str(e)
    supported=[k for k,v in contexts.items() if len(v)>=cfg['normal_dwell']['minimum_complete_runs']]
    if failure is None and not supported:failure='No supported normal entry context after role gate'
    support={k:{'runs':len(v),'distinct_videos':len(set(r['sequence'] for r in v)),'supported':k in supported,'observations':v} for k,v in contexts.items()}
    result={'scene':scene,'fit_only':not args.all_sequences,'normal_fit_support_passed':failure is None,'failure':failure,'sequences':rows,'qualitative_cases':qual,'contexts':support,'supported_contexts':supported,'model':getattr(model,'evidence',None),'interpretation':'Gate acceptance is not object accuracy. The six normal cases are qualitative checks; no test labels or margin tuning used.'}
    if not args.all_sequences:
        (out/'normal_anchor_audit.json').write_text(json.dumps(result,indent=2)+'\n')
        if transformed:
            target.mkdir(parents=True,exist_ok=True)
            for seq,d in transformed.items():np.savez_compressed(target/f'training_{seq}.npz',**d)
        if failure is None:np.savez_compressed('artifacts/experiment17/normal_relation_model.npz',location=model.location,scale=model.scale,centers=model.centers,area_roles=np.array(list(model.area_upper)),area_upper=np.array(list(model.area_upper.values())))
    else:
        if failure:raise ValueError(f'Frozen normal support failed: {failure}')
        normal=json.loads((out/'normal_anchor_audit.json').read_text());assert normal['normal_fit_support_passed']
        np.testing.assert_allclose(model.centers,np.array(normal['model']['cluster_centers_scaled']),rtol=0,atol=1e-12)
        target.mkdir(parents=True,exist_ok=True);records=[];hashes={}
        for path in sorted(source.glob('*.npz')):
            d=load(path);phase,valid,chosen,x=model.transform(d);margins=model.margins(d);derived=dict(d,phases=phase,relation_valid=valid,relation_detection_indices=chosen,relation_descriptors=x,anchor_margin=margins)
            np.savez_compressed(target/path.name,**derived)
            for key in d:
                if key not in ['phases','relation_valid','relation_detection_indices','relation_descriptors']:np.testing.assert_array_equal(d[key],derived[key])
            part,seq=path.stem.split('_');group='test' if part=='testing' else 'fit' if seq in split['fit'] else 'calibration'
            records.append({'sequence_key':path.stem,'group':group,'samples':len(valid),'valid_relations':int(valid.sum()),'anchor_boxes_before':int(np.sum(d['roles']==model.anchor_role)),'anchor_boxes_after':int(np.sum((d['roles']==model.anchor_role)&(margins>model.margin_threshold))),'phase_counts':np.bincount(phase,minlength=model.k).tolist()});hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        result.update(sequences=records,source_sha256=hashes);(out/'relation_features.json').write_text(json.dumps(result,indent=2)+'\n')
        old=json.loads(Path('results/experiment15/process_discovery.json').read_text());old.update(derived_from='results/experiment15/process_discovery.json',note='Normal object vocabulary retained; relation states refitted after fixed CLIP anchor verification. No new VLM generation and no semantic phase ground truth.')
        (out/'process_discovery.json').write_text(json.dumps(old,indent=2)+'\n')
    print(json.dumps({'normal_fit_support_passed':failure is None,'failure':failure,'supported_contexts':supported,'anchor_boxes_before':sum(r['anchor_boxes_before'] for r in rows),'anchor_boxes_after':sum(r['anchor_boxes_after'] for r in rows),'valid_relations_after':sum(r.get('valid_relations_after',0) for r in rows)},indent=2))


if __name__=='__main__':main()
