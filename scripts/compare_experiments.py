"""Paired descriptive comparison, with failure metrics included."""
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('results/comparison01_02');root.mkdir(parents=True,exist_ok=True)
data=[json.loads(Path(f'results/experiment{n}/metrics.json').read_text()) for n in ['01','02']]
rows=[]
for n,d in zip(['01','02'],data):
    rows.append({'experiment':n,'combined_auroc':d['metrics']['combined']['auroc'],
                 'combined_average_precision':d['metrics']['combined']['average_precision'],
                 'test_normal_frame_alarm_rate':d['test_normal_frame_alarm_rate'],
                 'test_anomaly_frame_recall_at_q99':d['test_anomaly_frame_recall_at_q99'],
                 'q99_threshold':d['normal_q99_threshold']})
with (root/'metrics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
summary={'scene':'R01','test_frames':3685,'test_sequences':15,'single_seed':42,
         'combined_auroc_delta':rows[1]['combined_auroc']-rows[0]['combined_auroc'],
         'combined_ap_delta':rows[1]['combined_average_precision']-rows[0]['combined_average_precision'],
         'confidence_interval':None,'interpretation':'Descriptive development-set result; increased AUROC/AP and recall, but increased false alarms. No statistical significance or final held-out performance claim.',
         'next_recommended_improvement':'Improve transported-product grounding and explicitly handle missing observations before extending process reasoning.',
         'next_experiment_status':'not implemented; no further experiment matrix planned'}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,keys,names in [(axes[0],['combined_auroc','combined_average_precision'],['AUROC','Average precision']),
                     (axes[1],['test_normal_frame_alarm_rate','test_anomaly_frame_recall_at_q99'],['False positive rate','Anomaly recall'])]:
    x=np.arange(2)
    for i,row in enumerate(rows):ax.bar(x+(i-.5)*.35,[row[k] for k in keys],.35,label=f'Experiment {row["experiment"]}')
    ax.set(xticks=x,xticklabels=names,ylim=(0,1),yticks=np.linspace(0,1,6));ax.legend()
axes[0].set_title('R01: combined score ranking')
axes[1].set_title('Each model at its normal calibration q99')
fig.savefig(root/'comparison.png',dpi=150)
