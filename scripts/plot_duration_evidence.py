"""Aggregate normal episode evidence; durations/censor bounds are kept distinct."""
import csv,json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/experiment32');r=json.loads((OUT/'summary.json').read_text());episodes=json.loads((OUT/'episodes.json').read_text())['episodes']
fig,axs=plt.subplots(1,2,figsize=(13,5.4));fig.subplots_adjust(left=.07,right=.98,bottom=.20,top=.81,wspace=.29)
x=np.arange(3)
for offset,part,color in [(-.18,'fit','#087f8c'),(.18,'calibration','#94a3b8')]:
 values=[r['partitions'][part]['total']['status'][k]['episodes'] for k in ['complete','right_censored','unknown_entry']];bars=axs[0].bar(x+offset,values,.34,color=color,label=part);axs[0].bar_label(bars,padding=3)
axs[0].set(xticks=x,xticklabels=['Complete','Right-censored','Unknown entry'],ylim=(0,260),ylabel='Episodes',title='A. All normal observations');axs[0].legend(frameon=False,loc='upper left')
x=np.arange(2);contexts=['1->3','3->1']
for offset,name,color in [(-.24,'Complete','#087f8c'),(0,'Positive censor bound','#d97706'),(.24,'Zero censor bound','#cbd5e1')]:
 values=[]
 for context in contexts:
  s=r['partitions']['fit']['contexts'][context];values.append(s['status']['complete']['episodes'] if name=='Complete' else s['zero_censor_bounds'] if name=='Zero censor bound' else s['status']['right_censored']['episodes']-s['zero_censor_bounds'])
 bars=axs[1].bar(x+offset,values,.22,color=color,label=name);axs[1].bar_label(bars,padding=3)
axs[1].set(xticks=x,xticklabels=contexts,ylim=(0,16),ylabel='FIT episodes with observed entry',title='B. Available duration evidence by context');axs[1].legend(frameon=False,loc='upper right',fontsize=9)
for ax in axs:ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
fig.suptitle('Experiment 32 | Identity-bounded normal duration evidence',fontsize=15,x=.07,ha='left')
fig.text(.07,.06,'Normal-only audit: no new anomaly scores. Unknown-entry spans are not duration labels.\nRight-censor bounds end at the last actual observation; they are not observed completion times.',fontsize=10,color='#475569')
fig.savefig(OUT/'duration_evidence.png',dpi=180);plt.close(fig)
rows=[];diagnostic={}
for part,partition in r['partitions'].items():
 selected=[x for x in episodes if x['partition']==part];unknown=[x for x in selected if x['status']=='unknown_entry'];diag={'unknown_entry_start_reasons':dict(Counter(x['start_reason'] for x in unknown)),'status_end_reasons':{state:dict(Counter(x['end_reason'] for x in selected if x['status']==state)) for state in ['complete','right_censored','unknown_entry']},'status_observed_samples':{state:sum(x['observed_samples'] for x in selected if x['status']==state) for state in ['complete','right_censored','unknown_entry']},'context_long_censor_bounds':{}}
 for context,s in partition['contexts'].items():
  rows.append({'partition':part,'context':context,'complete':s['status']['complete']['episodes'],'complete_videos':s['status']['complete']['videos'],'right_censored':s['status']['right_censored']['episodes'],'censored_videos':s['status']['right_censored']['videos'],'positive_censor_bounds':s['status']['right_censored']['episodes']-s['zero_censor_bounds'],'zero_censor_bounds':s['zero_censor_bounds'],'unknown_entry':s['status']['unknown_entry']['episodes'],'unknown_entry_videos':s['status']['unknown_entry']['videos'],'meets_existing_complete_support':s['meets_existing_complete_support']})
  if context.startswith('?'):continue
  current=[x for x in selected if f'{x["entry_context"]}->{x["phase"]}'==context];complete=[x['duration_frames'] for x in current if x['status']=='complete'];maximum=max(complete) if complete else None;long=[x for x in current if x['status']=='right_censored' and maximum is not None and x['censor_lower_bound_frames']>maximum];diag['context_long_censor_bounds'][context]={'max_observed_complete_duration':maximum,'censor_bounds_above_complete_max':len(long),'cases':[{'sequence':x['sequence'],'episode_id':x['episode_id'],'start_frame':x['first_observed_source_frame'],'lower_bound':x['censor_lower_bound_frames'],'end_reason':x['end_reason']} for x in long]}
 diagnostic[part]=diag
with (OUT/'context_evidence.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(OUT/'evidence_diagnostic.json').write_text(json.dumps({'normal_only':True,'partitions':diagnostic,'note':'Descriptive counts from frozen episode records. No new score/model fitting or test access.'},indent=2)+'\n')
