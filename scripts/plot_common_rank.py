"""Publication-friendly seed-level descriptive figures and exact comparison rows."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from audit_common_rank import VARIANTS,AFTER


def main():
    root=Path('results/experiment23');d=json.loads((root/'common_rank_diagnostic.json').read_text());h=json.loads((root/'normal_holdout.json').read_text());rows=[]
    labels=['Observed FIT','Random seed 0','Random seed 1','Random seed 2','Random seed 3','Random seed 4']
    for e in VARIANTS:
        v=d['variants'][e];rows.append({'variant':e,'normal_q99':v['q99'],'visual_auroc':v['metrics']['visual']['auroc'],'visual_ap':v['metrics']['visual']['average_precision'],'combined_auroc':v['metrics']['combined']['auroc'],'combined_ap':v['metrics']['combined']['average_precision'],'normal_holdout_fp':v['normal_holdout_fp'],'test_fp':v['alarms']['normal'],'test_fpr':v['normal_fpr'],'test_tp':v['alarms']['anomaly'],'test_recall':v['anomaly_recall'],'detected_events':v['events']['detected_events'],'conditional_median_delay_frames':v['events']['median_delay_detected_only_frames']})
    with (root/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    rows=[r for r in rows if r['variant'] in AFTER]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    colors=['#007F73',*['#345DA7']*5];markers=['D',*['o']*5];y=np.arange(6)
    fig,axes=plt.subplots(1,4,figsize=(16,5.3),sharey=True,gridspec_kw={'wspace':.30})
    for ax,key,title,bounds,fmt in zip(axes,['visual_auroc','combined_auroc','test_fp','detected_events'],['Visual AUROC','Combined AUROC','False-positive frames','Detected GT intervals'],[(.675,.709),(.66,.689),(290,425),(11,20)],['.4f','.4f','d','d']):
        for i,row in enumerate(rows):
            val=row[key];ax.scatter(val,i,s=65,c=colors[i],marker=markers[i],zorder=3);ax.annotate(format(val,fmt),(val,i),xytext=(6,6),textcoords='offset points',fontsize=10)
        ax.set_title(title,pad=14);ax.set_xlim(*bounds);ax.grid(axis='x',alpha=.18);ax.set_yticks(y,labels);ax.set_ylim(5.6,-.65)
        ax.set_xlabel('Score (0–1)' if 'auroc' in key else 'Normal denominator: 3,576' if key=='test_fp' else 'GT denominator: 26')
        if key=='detected_events':ax.set_xticks([12,14,16,18,20])
    fig.suptitle('Experiment 23 · Same FIT counts, support and PCA ranks',x=.03,ha='left',fontsize=17,fontweight='bold')
    fig.subplots_adjust(left=.14,right=.98,top=.82,bottom=.19)
    fig.text(.03,.05,'R04 development set: 19 test videos / 8,154 frames. Five fixed sampling seeds; points are not independent datasets or confidence intervals.\nEach configuration refits its normal CDF / q99. Source: common_rank_diagnostic.json; exact values: comparison.csv.',fontsize=10,color='#475467')
    fig.savefig(root/'common_rank_comparison.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
