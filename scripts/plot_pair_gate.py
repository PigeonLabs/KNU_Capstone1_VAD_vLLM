"""Experiment31 aggregate ranking and false-alarm comparisons."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/experiment31');s=json.loads((OUT/'diagnostic.json').read_text())['variants'];gates=['hold','pool','age'];colors=['#64748b','#087f8c'];fig,axs=plt.subplots(1,3,figsize=(16,5.3));fig.subplots_adjust(left=.06,right=.98,bottom=.19,top=.82,wspace=.40)
for j,gate in enumerate(gates):
 for offset,metric,marker in [(0,'auroc','o'),(.35,'average_precision','s')]:
  y=2-j+offset;vals=[s[f'{prefix}_{gate}']['metrics']['combined'][metric] for prefix in ['30_reset','31']];axs[0].plot(vals,[y,y],color='#cbd5e1',lw=2)
  for idx,x in enumerate(vals):
   axs[0].scatter(x,y,color=colors[idx],marker=marker,s=50);axs[0].annotate(f'{x:.4f}',(x,y),xytext=(-7 if idx==0 else 7,0),textcoords='offset points',ha='right' if idx==0 else 'left',va='center',fontsize=8)
axs[0].set(yticks=[.175,1.175,2.175],yticklabels=['age','pool','hold'],xlim=(.702,.725),ylim=(-.25,2.8),xlabel='Combined ranking (points)',title='A. Circles: AUROC; squares: AP')
x=np.arange(3)
for ax,key,title,denom in [(axs[1],'alarms','B. Test false positives',3576),(axs[2],'normal_holdout_fp','C. Normal-video holdout',1920)]:
 for idx,prefix in enumerate(['30_reset','31']):
  values=[s[f'{prefix}_{gate}']['alarms']['normal'] if key=='alarms' else s[f'{prefix}_{gate}'][key] for gate in gates];bars=ax.bar(x+[-.18,.18][idx],values,.34,color=colors[idx],label=prefix);ax.bar_label(bars,padding=3)
 ax.set(xticks=x,xticklabels=gates,ylim=(0,390 if key=='alarms' else 48),ylabel=f'False-positive frames / {denom:,}',title=title)
axs[2].legend(frameon=False,loc='upper right')
for ax in axs:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
fig.suptitle('Experiment 31 | Fuse transitions only across the same selected track pair',fontsize=15,x=.06,ha='left')
fig.text(.06,.045,'R04 development | Paired q99 unchanged | Test TP decreases by 4 frames in every route; detected GT intervals and conditional median delay unchanged',fontsize=9,color='#475569')
fig.savefig(OUT/'pair_gate_comparison.png',dpi=180);plt.close(fig)
rows=[]
for e,v in s.items():
 rows.append({'variant':e,'visual_auroc':v['metrics']['visual']['auroc'],'visual_ap':v['metrics']['visual']['average_precision'],'combined_auroc':v['metrics']['combined']['auroc'],'combined_ap':v['metrics']['combined']['average_precision'],'q99':v['q99'],'fp':v['alarms']['normal'],'tp':v['alarms']['anomaly'],'fpr':v['normal_fpr'],'recall':v['anomaly_recall'],'detected_events':v['events']['detected_events'],'events':26,'conditional_median_delay_frames':v['events']['median_delay_detected_only_frames'],'normal_holdout_fp':v['normal_holdout_fp'],'transition_gate_open_frames':v['transition_gate_open_frames'],'dwell_valid_frames':v['dwell_valid_frames']})
with (OUT/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
