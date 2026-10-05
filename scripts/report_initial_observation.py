"""Render experiment38 aggregate comparison and inspectable source tables."""
import csv,json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path('results/experiment38')
def read(n):return json.loads((OUT/n).read_text())
def write(n,v):(OUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    diagnostic=read('diagnostic.json');variants=diagnostic['variants'];rows=[]
    for e,v in variants.items():
        rows.append({'variant':e,'visual_auroc':v['metrics']['visual']['auroc'],'visual_ap':v['metrics']['visual']['average_precision'],'combined_auroc':v['metrics']['combined']['auroc'],'combined_ap':v['metrics']['combined']['average_precision'],'q99':v['q99'],'fp':v['alarms']['normal'],'tp':v['alarms']['anomaly'],'fpr':v['normal_fpr'],'recall':v['anomaly_recall'],'events_detected':v['events']['detected_events'],'events_total':26,'conditional_median_delay_frames':v['events']['median_delay_detected_only_frames'],'normal_holdout_fp':v['normal_holdout_fp']})
    with (OUT/'comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    contributions={}
    for e,v in variants.items():
        cells={};q=v['q99']
        for path in sorted(Path(f'artifacts/experiment{e}/predictions').glob('*.npz')):
            with np.load(path,allow_pickle=False) as archive:d=dict(archive)
            densephase=d['phases'][np.searchsorted(d['indices'],np.arange(len(d['labels'])),side='right')-1]
            only=d['dwell_valid']&(d['dwell']>q)&~(d['visual']>q)&~(d['transition_gated']>q)
            for entry,phase in set(zip(d['dwell_entry_context'][only],densephase[only])):
                mask=only&(d['dwell_entry_context']==entry)&(densephase==phase);key=f'{entry}->{phase}';cell=cells.setdefault(key,{'fp':0,'tp':0,'sequences':[]});cell['fp']+=int(np.sum(mask&(d['labels']==0)));cell['tp']+=int(np.sum(mask&(d['labels']==1)));cell['sequences'].append(path.stem)
        expected=diagnostic['branch_contribution'][e]['dwell_only_alarms']
        assert sum(v['fp'] for v in cells.values())==expected['normal'] and sum(v['tp'] for v in cells.values())==expected['anomaly']
        contributions[e]=cells
    write('dwell_only_contexts.json',contributions)
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.5));x=np.arange(3);colors={'control':'#325c86','guarded':'#be692a'}
    for ax,field,title in zip(axes,['combined_auroc','fp','tp'],['Combined AUROC','Normal false alarms (frames)','Anomaly alarms (frames)']):
        for j,group in enumerate(['control','guarded']):
            vals=[next(r[field] for r in rows if r['variant']==f'38_{group}_{gate}') for gate in ['hold','pool','age']];bars=ax.bar(x+(j-.5)*.36,vals,width=.34,color=colors[group],label='Original routing' if group=='control' else 'Initial observation guard')
            if field=='combined_auroc':
                for i,(bar,val) in enumerate(zip(bars,vals)):
                    gate=['hold','pool','age'][i];top=max(r[field] for r in rows if r['variant'].endswith('_'+gate));ax.text(bar.get_x()+bar.get_width()/2,top+.02+j*.06,f'{val:.4f}',ha='center',va='bottom',fontsize=9)
            else:ax.bar_label(bars,labels=[str(v) for v in vals],padding=3,fontsize=9)
        ax.set_xticks(x,['hold','pool','age']);ax.set_title(title,fontsize=11);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,1 if field=='combined_auroc' else max(r[field] for r in rows)*1.17)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.06),ncol=2,frameon=False);fig.suptitle('Experiment 38: pool appearance until the first observation',fontsize=15)
    fig.text(.5,.018,'R04 development test: 3,576 normal / 4,578 anomaly frames. Each model uses its own normal q99.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.06,right=.99,top=.82,bottom=.26,wspace=.24);fig.savefig(OUT/'initial_observation_comparison.png',dpi=180);plt.close(fig)
    print('Rendered initial-observation comparison:',len(rows),'variants')
if __name__=='__main__':main()
