"""Render experiment37 aggregate comparison and inspectable source tables."""
import csv,json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path('results/experiment37')
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
    normal=read('normal_transform.json')['rows'];test=read('test_transform.json')['rows'];episodes=read('normal_episodes.json')['episodes'];summary={}
    for part in ['fit','calibration','test']:
        rs=[r for r in normal+test if r['partition']==part];sub={k:sum(r[k] for r in rs) for k in ['samples','frames','anchor_selection_changed_samples','phase_changed_samples','phase_changed_frames','descriptor_changed_samples']};sub['groups']={}
        for group in ['control','asinh']:
            pc=Counter();ep=Counter();ph=np.zeros(4,int);obs=np.zeros(4,int)
            for r in rs:
                g=r['groups'][group];pc.update(g['pair_changes']);ep.update(g['episode_status']);ph+=g['phase_counts'];obs+=g['observed_phase_counts']
            context=Counter(str(e['entry_context'])+'->'+str(e['phase']) for e in episodes if e['partition']==part and e['group']==group and e['status']=='complete')
            sub['groups'][group]={'pair_changes':dict(pc),'episode_status':dict(ep),'all_phase_samples':ph.tolist(),'observed_phase_samples':obs.tolist(),'identity_complete_contexts':dict(context) if part!='test' else None}
        summary[part]=sub
    write('transform_summary.json',summary)
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.5));x=np.arange(3);colors={'control':'#325c86','asinh':'#be692a'}
    for ax,field,title in zip(axes,['combined_auroc','fp','tp'],['Combined AUROC','Normal false alarms (frames)','Anomaly alarms (frames)']):
        for j,group in enumerate(['control','asinh']):
            vals=[next(r[field] for r in rows if r['variant']==f'37_{group}_{gate}') for gate in ['hold','pool','age']];bars=ax.bar(x+(j-.5)*.36,vals,width=.34,color=colors[group],label='Linear (matched ranks)' if group=='control' else 'Asinh (matched ranks)')
            if field=='combined_auroc':
                for i,(bar,val) in enumerate(zip(bars,vals)):
                    gate=['hold','pool','age'][i];top=max(r[field] for r in rows if r['variant'].endswith('_'+gate));ax.text(bar.get_x()+bar.get_width()/2,top+.02+j*.06,f'{val:.4f}',ha='center',va='bottom',fontsize=9)
            else:ax.bar_label(bars,labels=[str(v) for v in vals],padding=3,fontsize=9)
        ax.set_xticks(x,['hold','pool','age']);ax.set_title(title,fontsize=11);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,1 if field=='combined_auroc' else max(r[field] for r in rows)*1.17)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.06),ncol=2,frameon=False);fig.suptitle('Experiment 37: compress extreme phase coordinates',fontsize=15)
    fig.text(.5,.018,'R04 development test: 3,576 normal / 4,578 anomaly frames. Each model uses its own normal q99.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.06,right=.99,top=.82,bottom=.26,wspace=.24);fig.savefig(OUT/'asinh_phase_comparison.png',dpi=180);plt.close(fig)
    phase=read('normal_phase_audit.json')['partitions'];fig,axes=plt.subplots(1,2,figsize=(10.4,4.9))
    for ax,part,title in zip(axes,['fit','calibration'],['Normal FIT (1,123 valid samples)','Normal calibration (297 valid samples)']):
        matrix=np.array(phase[part]['contingency_control_rows_asinh_columns']);ax.imshow(matrix,cmap='Blues',vmin=0,vmax=matrix.max())
        for i in range(4):
            for j in range(4):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()*.55 else '#243447',fontsize=13)
        ax.set_xticks(range(4));ax.set_yticks(range(4));ax.set_xlabel('Asinh phase ID');ax.set_ylabel('Control phase ID');ax.set_title(title,fontsize=12)
    fig.suptitle('Normal phase reassignment under unchanged object selection',fontsize=14)
    agreements=[phase[p]['agreement_under_frozen_fit_permutation'] for p in ['fit','calibration']];fig.text(.5,.035,f'Latent IDs, not action labels. Agreement under FIT mapping: {agreements[0]:.1%} / {agreements[1]:.1%}.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.06,right=.98,top=.84,bottom=.18,wspace=.25);fig.savefig(OUT/'normal_phase_contingency.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
