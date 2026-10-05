"""Readable exact-source tables; interpret the result separately in the report."""
import json
from pathlib import Path
import numpy as np
from prepare_representation40 import OUT


def table(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|',*['| '+' | '.join(map(str,row))+' |' for row in rows]])
def fmt(values,scale=1,digits=4):
    values=np.array(values)*scale
    return f'{values.mean():.{digits}f}'+(f' ± {values.std(ddof=1):.{digits}f}' if len(values)>1 else '')

def main():
    data=json.loads((OUT/'metrics.json').read_text());manifest=json.loads((OUT/'data_manifest.json').read_text());rows=[];out=[]
    for scene,s in manifest['subsplits'].items():
        metric=next(r for r in data['variants'] if r['scene']==scene);views=sum(sum(r['views'].values()) for r in manifest['sequences'] if r['scene']==scene)
        rows.append([scene,len(s['representation_train']),len(s['representation_validation']),len(s['normal_calibration']),views,metric['normal_frames'],metric['anomaly_frames'],metric['unknown_frames'],metric['events']['events']])
    out += ['## 데이터',table(['공정','표현 train','표현 val','정상 cal','정상 views','test 정상','test 이상','test 제외','이상 구간'],rows)]
    rows=[]
    for scene in ['R01','R02','R03','R04']:
        for arm in ['A','B','C','D']:
            rr=[r for r in data['variants'] if r['scene']==scene and r['run'].split('_')[0]==arm]
            rows.append([scene,arm,fmt([r['metrics']['visual']['auroc'] for r in rr]),fmt([r['metrics']['visual']['average_precision'] for r in rr]),fmt([r['metrics']['combined']['auroc'] for r in rr]),fmt([r['metrics']['combined']['average_precision'] for r in rr]),fmt([r['normal_fpr'] for r in rr],100,2),fmt([r['anomaly_recall'] for r in rr],100,2),fmt([r['events']['event_coverage'] for r in rr],100,2)])
    out+=['## 공정별 탐지',table(['공정','arm','Visual AUC','Visual AP','Combined AUC','Combined AP','FPR%','recall%','구간 탐지%'],rows)]
    rows=[]
    for arm in ['A','B','C','D']:
        rr=[r for r in data['scene_macro'] if r['run'].split('_')[0]==arm]
        rows.append([arm,*[fmt([r[k] for r in rr],100 if k in ['normal_fpr','anomaly_recall','event_coverage'] else 1,2 if k in ['normal_fpr','anomaly_recall','event_coverage'] else 4) for k in ['visual_auroc','combined_auroc','combined_average_precision','normal_fpr','anomaly_recall','event_coverage']]])
    out+=['## 공정 macro',table(['arm','Visual AUC','Combined AUC','Combined AP','FPR%','recall%','구간 탐지%'],rows)]
    rows=[]
    for seed in [42,43,44]:
        for arm in ['C','D']:
            run=f'{arm}_s{seed}';r=json.loads((OUT/f'{run}_training.json').read_text());h=json.loads((OUT/f'{run}_history.json').read_text());best=h[r['selected_epoch']]
            rows.append([run,r['scope']['trainable_parameters'],r['selected_epoch'],r['optimizer_steps'],f"{best['train']['loss']:.5f}" if best['train'] else '—',f"{best['validation']['loss']:.5f}",f"{best['validation']['teacher_cosine_loss']:.5f}",f"{best['validation']['effective_rank']:.2f}",f"{r['elapsed_seconds']:.1f}",f"{r['gpu_peak_allocated_bytes']/2**30:.2f}"])
    out+=['## 학습',table(['run','학습 파라미터','선택 epoch','updates','train loss','val loss','teacher 거리','유효 rank','초','GPU peak GiB'],rows)]
    rows=[]
    for r in data['variants']:
        ranks=[s['rank'] for s in r['subspaces']];rows.append([r['scene'],r['run'],f"{r['q99']:.8f}",r['normal_holdout_fp'],r['normal_holdout_frames'],r['fp'],r['tp'],f"{r['events']['detected_events']}/{r['events']['events']}",f'{min(ranks)}–{max(ranks)}'])
    out+=['## run별 임계값·오탐·구간',table(['공정','run','정상 q99','정상 LOVO FP','LOVO frames','test FP','test TP','탐지 구간','PCA rank 범위'],rows)]
    diagnostic=json.loads((OUT/'representation_diagnostic.json').read_text());rows=[]
    for scene in ['R01','R02','R03','R04']:
        for kind in ['global','crop']:
            rr=[r for r in diagnostic['normal_features'] if r['scene']==scene and r['partition']=='normal_calibration' and r['kind']==kind]
            values=[fmt([r['sample_effective_rank'] for r in rr if r['run'].split('_')[0]==arm],digits=2) for arm in ['A','B','C','D']]
            drift=[fmt([r['cosine_distance_from_B_mean'] for r in rr if r['run'].split('_')[0]==arm],digits=4) for arm in ['C','D']]
            rows.append([scene,kind,*values,*drift])
    out+=['## 정상 calibration 표현 진단',table(['공정','view','A rank','B rank','C rank','D rank','C→B 거리','D→B 거리'],rows)]
    (OUT/'tables.md').write_text('\n\n'.join(out)+'\n')

if __name__=='__main__':main()
