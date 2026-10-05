"""Reproducible report tables; require the completed independent validation record."""
import json
from pathlib import Path
import numpy as np
from ipad_vad.learned_detector import sha,write

OUT=Path('results/experiment41')
SCENES=['R01','R02','R03','R04'];ARMS=['A','B','C','D']

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def fmt(values,percent=False):
    a=np.array(values)*(100 if percent else 1);digits=2 if percent else 4
    s=f'{a.mean():.{digits}f}'
    return s+(f' ± {a.std(ddof=1):.{digits}f}' if len(a)>1 else '')

def main():
    validation=json.loads((OUT/'validation.json').read_text());assert validation['test_predictions_reconstructed']==528
    data={i:json.loads(Path(f'results/experiment{i}/metrics.json').read_text()) for i in [40,41]};diag=json.loads((OUT/'normal_diagnostics.json').read_text());parts=[];rows=[]
    for arm in ARMS:
        for exp in [40,41]:
            a=[r for r in data[exp]['scene_macro'] if r['run'].split('_')[0]==arm]
            rows.append([exp,arm,*[fmt([v[k] for v in a],k in ['normal_fpr','anomaly_recall','event_coverage']) for k in ['visual_auroc','combined_auroc','combined_average_precision','normal_fpr','anomaly_recall','event_coverage']]])
    parts+=['## Macro comparison',table(['exp','arm','Visual AUC','Combined AUC','Combined AP','FPR%','recall%','event%'],rows),'Within-process metrics equally averaged. C/D mean ± prior visual seed sample SD, not detector seed uncertainty. Separate normal q99 per run/process. Experiment41 R04 adds explicit unavailable-dwell handling after normal fit failed; see normal_fit_recovery.json.']
    rows=[]
    for s in SCENES:
        for arm in ARMS:
            for exp in [40,41]:
                a=[r for r in data[exp]['variants'] if r['scene']==s and r['run'].split('_')[0]==arm]
                rows.append([s,arm,exp,fmt([v['metrics']['visual']['auroc'] for v in a]),fmt([v['metrics']['combined']['auroc'] for v in a]),fmt([v['metrics']['combined']['average_precision'] for v in a]),fmt([v['normal_fpr'] for v in a],True),fmt([v['anomaly_recall'] for v in a],True),fmt([v['events']['event_coverage'] for v in a],True),', '.join(str(v['normal_holdout_fp'])+'/'+str(v['normal_holdout_frames']) for v in a)])
    parts+=['## Per-process comparison',table(['process','arm','exp','Visual AUC','Combined AUC','Combined AP','FPR%','recall%','event%','normal LOVO FP/frames'],rows)]
    rows=[];rollup=[]
    for s in SCENES:
        for a in ['frozen','learned']:
            for role in range(2 if s=='R03' else 3):
                v=[r for r in diag['per_video_role'] if r['scene']==s and r['arm']==a and r['role']==role];n=sum(r['samples'] for r in v);observed=sum(r['observed'] for r in v);tracks=sum(r['tracks'] for r in v);missing=max(r['max_missing_samples'] for r in v)
                row={'scene':s,'arm':a,'role':role,'sampled_frames':n,'observed_frames':observed,'coverage':observed/n,'sum_track_ids_within_videos':tracks,'maximum_missing_samples':missing};rollup.append(row)
                rows.append([s,a,role,f'{observed}/{n}',f'{observed/n*100:.2f}',tracks,missing])
    parts+=['## Normal observation diagnostics',table(['process','detector','role','observed/samples','coverage%','sum track IDs','longest gap (samples)'],rows),'Observation coverage is not detector recall. Fewer track IDs may mean less fragmentation, fewer objects or wrong merges; no ID ground truth. Sampling stride4; no seconds inferred.']
    rows=[]
    for r in diag['weak_agreement_summary']:rows.append([r['scene'],r['arm'],r['role'],r['matched_iou_ge05'],r['weak_target_boxes'],r['predicted_boxes']])
    parts+=['## Weak normal-validation agreement',table(['process','detector','role','matched IoU>=0.5','weak targets','predictions'],rows),'Not independent detection accuracy: approximate validation boxes also selected the checkpoint.']
    text='\n\n'.join(parts)+'\n';(OUT/'report_tables.md').write_text(text)
    write(OUT/'report_summary.json',{'normal_observation_rollup':rollup,'source_sha256':{str(p):sha(p) for p in [Path('results/experiment40/metrics.json'),OUT/'metrics.json',OUT/'validation.json',OUT/'normal_diagnostics.json']},'tables_sha256':sha(OUT/'report_tables.md')})

if __name__=='__main__':main()
