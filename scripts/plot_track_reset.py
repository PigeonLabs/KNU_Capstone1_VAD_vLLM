"""Aggregate experiment30 figures; never publishes source images/features."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/experiment30')
r=json.loads((OUT/'diagnostic.json').read_text());s=r['variants'];gates=['hold','pool','age'];colors={'control':'#64748b','reset':'#087f8c'}
fig,axs=plt.subplots(1,3,figsize=(16,5.3));fig.subplots_adjust(left=.06,right=.98,bottom=.19,top=.82,wspace=.40)
for j,gate in enumerate(gates):
 for offset,metric,marker in [(0,'auroc','o'),(.35,'average_precision','s')]:
  y=2-j+offset;vals=[s[f'30_{g}_{gate}']['metrics']['combined'][metric] for g in ['control','reset']]
  axs[0].plot(vals,[y,y],color='#cbd5e1',lw=2)
  for g,x in zip(['control','reset'],vals):
   axs[0].scatter(x,y,color=colors[g],marker=marker,s=50);axs[0].annotate(f'{x:.4f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
axs[0].set(yticks=[.175,1.175,2.175],yticklabels=['age','pool','hold'],xlim=(.668,.73),ylim=(-.25,2.8),xlabel='Combined score ranking (points)',title='A. Ranking: circles AUROC, squares AP')
for gate,marker in zip(gates,['o','s','^']):
 a,b=[s[f'30_{g}_{gate}']['alarms'] for g in ['control','reset']]
 axs[1].plot([a['normal'],b['normal']],[a['anomaly'],b['anomaly']],color='#cbd5e1',lw=1.5)
 for group,x in [('control',a),('reset',b)]:
  axs[1].scatter(x['normal'],x['anomaly'],c=colors[group],marker=marker,s=55)
  dy={'hold':-15,'pool':-12,'age':8}[gate]
  axs[1].annotate(f'{gate} {x["normal"]}/{x["anomaly"]}',(x['normal'],x['anomaly']),xytext=(-8 if group=='reset' else 7,dy),ha='right' if group=='reset' else 'left',textcoords='offset points',fontsize=8)
axs[1].set(xlim=(240,346),ylim=(505,893),xlabel='False-positive frames / 3,576',ylabel='True-positive frames / 4,578',title='B. Alarms at each normal q99')
x=np.arange(3)
for offset,group in [(-.18,'control'),(.18,'reset')]:
 values=[s[f'30_{group}_{g}']['normal_holdout_fp'] for g in gates];bars=axs[2].bar(x+offset,values,.34,label=group,color=colors[group]);axs[2].bar_label(bars,padding=3)
axs[2].set(xticks=x,xticklabels=gates,ylim=(0,48),ylabel='False-positive frames / 1,920',title='C. Leave-one-normal-video-out')
for ax in axs:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.13);ax.set_axisbelow(True)
axs[2].legend(loc='upper right',frameon=False)
fig.suptitle('Experiment 30 | Reset relation history at selected track-pair changes',fontsize=15,x=.06,ha='left')
fig.text(.06,.045,'R04 development only | Fixed phase centers; refitted normal score models | Same phase-bank ranks | Own q99; strict >; causal frame hold',fontsize=10,color='#475569')
fig.savefig(OUT/'track_reset_comparison.png',dpi=180);plt.close(fig)
rows=[]
for e,v in s.items():
 rows.append({'variant':e,'visual_auroc':v['metrics']['visual']['auroc'],'visual_ap':v['metrics']['visual']['average_precision'],'combined_auroc':v['metrics']['combined']['auroc'],'combined_ap':v['metrics']['combined']['average_precision'],'normal_q99':v['q99'],'fp':v['alarms']['normal'],'tp':v['alarms']['anomaly'],'fpr':v['normal_fpr'],'recall':v['anomaly_recall'],'detected_events':v['events']['detected_events'],'events':26,'conditional_median_delay_frames':v['events']['median_delay_detected_only_frames'],'normal_holdout_fp':v['normal_holdout_fp'],'dwell_valid_frames':v['dwell_valid_frames']})
with (OUT/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
