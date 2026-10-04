"""Experiment27 paired performance and process-only alarm attribution."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_transition_gate import VARIANTS,AFTER,PAIRS


def main():
    root=Path('results/experiment27');d=json.loads((root/'transition_gate_diagnostic.json').read_text());rows=[]
    for e in VARIANTS:
        v=d['variants'][e];m=v['metrics'];rows.append({'variant':e,'normal_q99':v['q99'],'visual_auroc':m['visual']['auroc'],'visual_ap':m['visual']['average_precision'],'combined_auroc':m['combined']['auroc'],'combined_ap':m['combined']['average_precision'],'normal_holdout_fp':v['normal_holdout_fp'],'test_fp':v['alarms']['normal'],'test_fpr':v['normal_fpr'],'test_tp':v['alarms']['anomaly'],'test_recall':v['anomaly_recall'],'detected_events':v['events']['detected_events'],'conditional_median_delay_frames':v['events']['median_delay_detected_only_frames']})
    with (root/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    lookup={r['variant']:r for r in rows};names=['Hold','Immediate pooled','Age limit: 56 frames'];colors=['#667085','#007F73'];plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,key,title,bounds,fmt in zip(axes,['combined_auroc','test_fp'],['Combined AUROC','Normal false-positive frames'],[(.658,.691),(235,335)],['.4f','d']):
        for i,e in enumerate(AFTER):
            a,b=lookup[PAIRS[e]][key],lookup[e][key];ax.plot([a,b],[i,i],c='#B0BAC5',lw=2)
            for value,color,marker,label,offset in [(a,colors[0],'o','26: all transitions',12),(b,colors[1],'D','27: observed pairs',-21)]:
                ax.scatter(value,i,c=color,marker=marker,s=65,label=label if i==0 else None);ax.annotate(format(value,fmt),(value,i),xytext=(0,offset),textcoords='offset points',ha='center',color=color)
        ax.set_title(title);ax.set_yticks(range(3),names);ax.set_ylim(2.6,-.6);ax.set_xlim(*bounds);ax.grid(axis='x',alpha=.18);ax.set_xlabel('AUROC (0–1)' if key=='combined_auroc' else 'Normal denominator: 3,576 frames')
    fig.suptitle('Experiment 27 · Fuse only consecutively observed transitions',x=.02,ha='left',fontsize=15,fontweight='bold');h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.55,.91),ncol=2,frameon=False);fig.subplots_adjust(left=.18,right=.98,top=.75,bottom=.24,wspace=.22);fig.text(.02,.05,'R04: 19 development videos / 8,154 frames. Paired q99 unchanged; each route loses 16 FP and 4 TP frames.\nDescriptive comparisons; no confidence interval or independent generalization claim. Source: comparison.csv.',fontsize=9,color='#475467');fig.savefig(root/'transition_gate_comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8));x=np.arange(2);width=.32
    for offset,e,color,label in [(-width/2,'26_hold',colors[0],'26: all transitions'),(width/2,'27_hold',colors[1],'27: observed pairs')]:
        vals=[d['branch_contribution'][e]['process_added_over_visual'][k] for k in ['normal','anomaly']];bars=ax.bar(x+offset,vals,width,color=color,label=label);ax.bar_label(bars,padding=5)
    ax.set_xticks(x,['Normal FP added\n(normal denominator: 3,576)','Anomaly TP added\n(anomaly denominator: 4,578)']);ax.set_ylabel('Frames added beyond visual alarms');ax.set_ylim(0,42);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.set_title('Process-only alarm contribution halves; no new detected interval',loc='left',fontweight='bold',pad=20);ax.legend(frameon=False);fig.subplots_adjust(left=.10,right=.97,top=.83,bottom=.28);fig.text(.02,.05,'Counts identical for hold / pool / age. All added alarms come from transition; dwell adds zero.\nEach comparison uses that route’s combined normal q99 (not a separately fitted visual threshold).\nSource: transition_gate_diagnostic.json, branch_contribution.',fontsize=9,color='#475467');fig.savefig(root/'process_contribution.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
