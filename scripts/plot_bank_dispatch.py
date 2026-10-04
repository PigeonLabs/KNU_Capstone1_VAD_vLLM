"""Experiment26 paired ranking and implementation-invariance evidence."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_bank_dispatch import VARIANTS,AFTER,PAIRS


def main():
    root=Path('results/experiment26');d=json.loads((root/'bank_dispatch_diagnostic.json').read_text());rows=[]
    for e in VARIANTS:
        v=d['variants'][e];m=v['metrics'];rows.append({'variant':e,'normal_q99':v['q99'],'visual_auroc':m['visual']['auroc'],'visual_ap':m['visual']['average_precision'],'combined_auroc':m['combined']['auroc'],'combined_ap':m['combined']['average_precision'],'normal_holdout_fp':v['normal_holdout_fp'],'test_fp':v['alarms']['normal'],'test_fpr':v['normal_fpr'],'test_tp':v['alarms']['anomaly'],'test_recall':v['anomaly_recall'],'detected_events':v['events']['detected_events'],'conditional_median_delay_frames':v['events']['median_delay_detected_only_frames']})
    with (root/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    lookup={r['variant']:r for r in rows};names=['Hold','Immediate pooled','Age limit: 56 frames'];colors=['#667085','#007F73']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,key,title in zip(axes,['visual_auroc','combined_auroc'],['Visual AUROC','Combined AUROC']):
        for i,e in enumerate(AFTER):
            a,b=lookup[PAIRS[e]][key],lookup[e][key];ax.plot([a,b],[i,i],color='#B0BAC5',lw=2)
            ax.scatter(a,i,c=colors[0],marker='o',s=65,label='25: request dispatch' if i==0 else None);ax.scatter(b,i,c=colors[1],marker='D',s=60,label='26: actual-bank dispatch' if i==0 else None)
            ax.annotate(f'{a:.4f}',(a,i),xytext=(0,12),textcoords='offset points',ha='center',color=colors[0]);ax.annotate(f'{b:.4f}',(b,i),xytext=(0,-21),textcoords='offset points',ha='center',color=colors[1])
        ax.set_title(title);ax.set_yticks(range(3),names);ax.set_ylim(2.6,-.6);ax.set_xlim(.655,.708);ax.grid(axis='x',alpha=.2);ax.set_xlabel('AUROC (0–1); descriptive single-scene comparison')
    fig.suptitle('Experiment 26 · Same banks and references, different CDF dispatch',x=.02,ha='left',fontsize=15,fontweight='bold');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,.91),ncol=2,frameon=False);fig.subplots_adjust(left=.18,right=.98,top=.75,bottom=.24,wspace=.15)
    fig.text(.02,.05,'R04: 19 development test videos / 8,154 frames. Paired q99 thresholds and all alarms are unchanged.\nNo uncertainty interval or independent generalization claim. Source: bank_dispatch_diagnostic.json / comparison.csv.',fontsize=9,color='#475467');fig.savefig(root/'bank_dispatch_comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8));pairs=[('hold','pool'),('hold','age'),('pool','age')];x=list(range(3))
    for offset,exp,color,label in [(-.18,'25',colors[0],'25: request dispatch'),(.18,'26',colors[1],'26: actual-bank dispatch')]:
        vals=[d['same_bank_different_request'][f'{exp}_{a}_to_{exp}_{b}']['changed_calibrated_scores'] for a,b in pairs];bars=ax.bar([i+offset for i in x],vals,width=.34,color=color,label=label);ax.bar_label(bars,padding=4)
    ax.set_xticks(x,[f'{a} vs {b}\n(n={d["same_bank_different_request"][f"26_{a}_to_26_{b}"]["feature_observations"]})' for a,b in pairs]);ax.set_ylim(0,620);ax.set_ylabel('Unequal calibrated feature-score comparisons');ax.set_title('Same actual bank, different requested route',loc='left',fontweight='bold',pad=18);ax.legend(frameon=False,loc='upper right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.subplots_adjust(left=.12,right=.97,top=.85,bottom=.26)
    fig.text(.02,.05,'R04 test global/crop observations. Counts are pairwise comparisons, not unique frames.\nBoth full-normal reference arrays remain unchanged. Source: bank_dispatch_diagnostic.json.',fontsize=9,color='#475467');fig.savefig(root/'same_bank_consistency.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
