"""Replace only phase assignments using normal FIT relation clustering."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ipad_vad.experiment import load_process
from ipad_vad.relational_phase import RelationalPhase


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/experiment08.json'));p.add_argument('--fit-only',action='store_true');args=p.parse_args()
    cfg=json.loads(args.config.read_text());scene=cfg['scene'];base=json.loads(Path(f'configs/experiment{cfg["feature_source_experiment"]}.json').read_text());original_process=load_process(base)
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    source=Path('artifacts')/('experiment'+cfg['feature_source_experiment'])/'features'/scene
    target=Path('artifacts')/('experiment'+cfg['experiment'])/'features'/scene
    normal=[load(source/f'training_{s}.npz') for s in split['fit']]
    model=RelationalPhase(**cfg['relational_phase'])
    with threadpool_limits(limits=4):model.fit(normal)
    paths=[source/f'training_{s}.npz' for s in split['fit']] if args.fit_only else sorted(source.glob('*.npz'))
    rows=[];hashes={}
    for path in paths:
        d=load(path);phases,valid,chosen,x=model.transform(d)
        part,seq=path.stem.split('_');group='test' if part=='testing' else 'fit' if seq in split['fit'] else 'calibration'
        rows.append({'sequence_key':path.stem,'group':group,'samples':len(valid),'valid_relations':int(valid.sum()),
                     'phase_counts':np.bincount(phases,minlength=model.k).tolist(),'changed_phase_samples':int((phases!=d['phases']).sum()),
                     'phase_transitions':int((phases[1:]!=phases[:-1]).sum())})
        if not args.fit_only:
            original={k:v for k,v in d.items()};d.update(phases=phases,relation_valid=valid,relation_detection_indices=chosen,relation_descriptors=x)
            target.mkdir(parents=True,exist_ok=True);np.savez_compressed(target/path.name,**d);check=load(target/path.name)
            assert all(np.array_equal(original[k],check[k]) for k in original if k!='phases'),path
            hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    groups={}
    for group in ('fit','calibration','test'):
        selected=[r for r in rows if r['group']==group]
        if selected:
            n=sum(r['samples'] for r in selected);v=sum(r['valid_relations'] for r in selected)
            groups[group]={'samples':n,'valid_relations':v,'valid_fraction':v/n,'phase_counts':np.sum([r['phase_counts'] for r in selected],axis=0).tolist()}
    result={'scene':scene,'fit_only':args.fit_only,'model':model.evidence,'groups':groups,'sequences':rows,'source_sha256':hashes,
            'source_arrays_preserved_except_phases':not args.fit_only,'missing_policy':'Hold last phase, initial latent_0; mark invalid and clear descriptor smoothing history. Invalid sentinel excluded from clustering.',
            'limitations':['No independent object/phase GT.','Normal temporal ordering is a heuristic, not semantic supervision.','Area gates affect phase anchors only; appearance boxes remain unchanged.']}
    out=Path('results')/('experiment'+cfg['experiment']);out.mkdir(parents=True,exist_ok=True)
    (out/('normal_relation_diagnostic.json' if args.fit_only else 'relation_features.json')).write_text(json.dumps(result,indent=2)+'\n')
    if not args.fit_only:
        source_record=json.loads(Path(base['process_discovery']).read_text())
        record={'scene':scene,'sources':source_record['sources'],'derived_from':base['process_discovery']}
        record['process']={'objects':original_process['objects'],'phases':[{'id':f'latent_{i}','description':f'Normal relation cluster {i}; no verified action label.'} for i in range(model.k)],
            'normal_order':[f'latent_{i}' for i in range(model.k)],'cyclic':original_process['cyclic'],'uncertainties':['Cluster order uses normal median relative position; original four-state self/next/cycle scoring template retained.']}
        record['note']='Derived normal relation states, not a new VLM generation. Sources refer to original normal object vocabulary discovery.'
        Path(cfg['process_discovery']).write_text(json.dumps(record,indent=2)+'\n')
        np.savez_compressed(target.parent.parent/'relation_model.npz',location=model.location,scale=model.scale,centers=model.centers,
                            area_roles=np.array(list(model.area_upper)),area_upper=np.array(list(model.area_upper.values())))
    print(json.dumps({'model':model.evidence,'groups':groups},indent=2))

if __name__=='__main__':main()
