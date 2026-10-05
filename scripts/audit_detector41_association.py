"""Normal-only inventory of possible identity supervision; not identity labels."""
import json
from pathlib import Path
import numpy as np
from ipad_vad.tracking import iou_matrix
from ipad_vad.learned_detector import write
from experiment41_phases import load_npz

def main():
    m=json.loads(Path('results/experiment40/data_manifest.json').read_text());rows=[]
    for scene,split in m['subsplits'].items():
        for group in ['representation_train','representation_validation','normal_calibration']:
            for seq in split[group]:
                d=load_npz(Path('artifacts/experiment41/detections/features')/scene/f'training_{seq}.npz')
                for role in range(2 if scene=='R03' else 3):
                    n=len(d['indices']);positive=negative=ambiguous=0;ids_by_frame=[np.flatnonzero((d['roles']==role)&(d['object_frames']==t)&(d['confidence']>=.5)) for t in range(n)]
                    for t,ids in enumerate(ids_by_frame):
                        if len(ids)>1:
                            overlap=iou_matrix(d['boxes'][ids],d['boxes'][ids]);negative+=int(np.sum(np.triu(overlap<=.1,k=1)))
                        if not t or not len(ids) or not len(ids_by_frame[t-1]):continue
                        prev=ids_by_frame[t-1];overlap=iou_matrix(d['boxes'][prev],d['boxes'][ids]);edges=overlap>=.7
                        positive+=sum(bool(edges[i,j] and edges[i].sum()==1 and edges[:,j].sum()==1) for i,j in zip(*np.where(edges)))
                        ambiguous+=int(np.sum(overlap>=.2)-np.sum(edges))
                    rows.append({'scene':scene,'group':group,'sequence':seq,'role':role,'unique_adjacent_iou_ge07_candidates':int(positive),'same_frame_iou_le01_candidates':negative,'adjacent_iou_02_to_07_edges':ambiguous})
    write('results/experiment41/association_supervision_inventory.json',{'normal_only':True,'confidence_min':.5,'positive_rule':'unique row/column adjacent IoU >= .7, candidate only','negative_rule':'same-frame same-role distinct boxes IoU <= .1, candidate only','identity_ground_truth':False,'rows':rows,'limitations':'Boxes can split one object or capture false positives; high IoU can switch identities. No cross-video identities or human labels inferred. Counts cannot establish ReID accuracy.'})

if __name__=='__main__':main()
