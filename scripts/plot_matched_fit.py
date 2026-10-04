"""Publication-friendly seed-level descriptive figures and exact comparison rows."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from audit_matched_fit import VARIANTS


def main():
    root=Path('results/experiment22');d=json.loads((root/'matched_fit_diagnostic.json').read_text());h=json.loads((root/'normal_holdout.json').read_text());rows=[]
    labels=['All FIT (18)','Observed FIT (21)','Random seed 0','Random seed 1','Random seed 2','Random seed 3','Random seed 4']
    for e in VARIANTS:
        v=d['variants'][e];rows.append({'variant':e,'normal_q99':v['q99'],'visual_auroc':v['metrics']['visual']['auroc'],'visual_ap':v['metrics']['visual']['average_precision'],'combined_auroc':v['metrics']['combined']['auroc'],'combined_ap':v['metrics']['combined']['average_precision'],'normal_holdout_fp':v['normal_holdout_fp'],'test_fp':v['alarms']['normal'],'test_fpr':v['normal_fpr'],'test_tp':v['alarms']['anomaly'],'test_recall':v['anomaly_recall'],'detected_events':v['events']['detected_events'],'conditional_median_delay_frames':v['events']['median_delay_detected_only_frames']})
    with (root/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    colors=['#667085','#007F73',*['#345DA7']*5];markers=['s','D',*['o']*5];y=np.arange(7)
    fig,axes=plt.subplots(1,4,figsize=(16,5.3),sharey=True,gridspec_kw={'wspace':.30})
    for ax,key,title,bounds,fmt in zip(axes,['visual_auroc','combined_auroc','test_fp','detected_events'],['Visual AUROC','Combined AUROC','False-positive frames','Detected GT intervals'],[(.675,.709),(.66,.689),(290,425),(11,20)],['.4f','.4f','d','d']):
        for i,row in enumerate(rows):
            val=row[key];ax.scatter(val,i,s=65,c=colors[i],marker=markers[i],zorder=3);ax.annotate(format(val,fmt),(val,i),xytext=(6,6),textcoords='offset points',fontsize=10)
        ax.set_title(title,pad=14);ax.set_xlim(*bounds);ax.grid(axis='x',alpha=.18);ax.set_yticks(y,labels);ax.set_ylim(6.6,-.65)
        ax.set_xlabel('Score (0–1)' if 'auroc' in key else 'Normal denominator: 3,576' if key=='test_fp' else 'GT denominator: 26')
        if key=='detected_events':ax.set_xticks([12,14,16,18,20])
    fig.suptitle('Experiment 22 · Count-matched normal FIT controls',x=.03,ha='left',fontsize=17,fontweight='bold')
    fig.subplots_adjust(left=.14,right=.98,top=.82,bottom=.19)
    fig.text(.03,.05,'R04 development set: 19 test videos / 8,154 frames. Five fixed sampling seeds; points are not independent datasets or confidence intervals.\nEach configuration refits its normal CDF / q99. Source: matched_fit_diagnostic.json; exact values: comparison.csv.',fontsize=10,color='#475467')
    fig.savefig(root/'matched_fit_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.6),gridspec_kw={'width_ratios':[1.25,1]})
    matrix=np.array([[next(r['held_out_frame_alarms'] for r in h['folds'] if r['variant']==e and r['held_out_sequence']==s) for s in h['holdout_sequences']] for e in VARIANTS]);ax=axes[0];im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(32,matrix.max()),aspect='auto');ax.set_xticks(range(5),h['holdout_sequences']);ax.set_yticks(y,labels);ax.set_xlabel('Excluded normal calibration video');ax.set_title('Normal holdout false-positive frames')
    for i in range(7):
        for j in range(5):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>19 else '#101828')
    fig.colorbar(im,ax=ax,fraction=.035,pad=.03,label='Frames')
    ax=axes[1]
    for i,e in enumerate(VARIANTS):
        for v,marker,offset in [('False','o',-.14),('True','s',.14)]:
            value=d['strata_relation_observed'][e][v]['metrics']['visual']['auroc'];ax.scatter(value,i+offset,c=colors[i],marker=marker,s=45,label=('Missing relation' if v=='False' else 'Observed relation') if i==0 else None)
    ax.set_yticks(y,labels);ax.set_ylim(6.6,-.6);ax.set_xlim(.60,.84);ax.set_xlabel('Visual AUROC (0–1)');ax.set_title('Fixed relation-observation subsets');ax.grid(axis='x',alpha=.18)
    fig.suptitle('Normal calibration sensitivity and test subsets',fontsize=16,fontweight='bold',x=.02,ha='left');fig.subplots_adjust(left=.16,right=.97,top=.85,bottom=.2,wspace=.57)
    fig.text(.02,.05,'Normal holdout: 5 videos / 1,920 frames. Test observed: 5,599 frames; missing: 2,555 frames.\nCircles: missing; squares: observed. Identical subsets. Sources: normal_holdout.json / matched_fit_diagnostic.json.',fontsize=10,color='#475467')
    fig.savefig(root/'holdout_and_subsets.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
