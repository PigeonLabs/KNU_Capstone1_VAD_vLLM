"""Experiment24 evidence-backed comparison and age-stratified alarms."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_missing_age import STATE_NAMES
from audit_request_calibration import VARIANTS,AFTER


def main():
    root=Path('results/experiment25');d=json.loads((root/'request_calibration_diagnostic.json').read_text());rows=[];names=['Unlimited hold','Immediate pooled','Age limit: 56 frames'];colors=['#667085','#345DA7','#007F73']
    for e in VARIANTS:
        v=d['variants'][e];m=v['metrics'];rows.append({'variant':e,'normal_q99':v['q99'],'visual_auroc':m['visual']['auroc'],'visual_ap':m['visual']['average_precision'],'combined_auroc':m['combined']['auroc'],'combined_ap':m['combined']['average_precision'],'normal_holdout_fp':v['normal_holdout_fp'],'test_fp':v['alarms']['normal'],'test_fpr':v['normal_fpr'],'test_tp':v['alarms']['anomaly'],'test_recall':v['anomaly_recall'],'detected_events':v['events']['detected_events'],'conditional_median_delay_frames':v['events']['median_delay_detected_only_frames']})
    with (root/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    rows=[r for r in rows if r['variant'] in AFTER]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    fig,axes=plt.subplots(1,4,figsize=(15,4.7),sharey=True)
    for ax,key,title,bounds,fmt in zip(axes,['visual_auroc','combined_auroc','test_fp','detected_events'],['Visual AUROC','Combined AUROC','False-positive frames','Detected GT intervals'],[(.68,.715),(.65,.70),(250,340),(10,18)],['.4f','.4f','d','d']):
        for i,row in enumerate(rows):
            v=row[key];ax.scatter(v,i,s=80,c=colors[i],marker=['s','o','D'][i]);ax.annotate(format(v,fmt),(v,i),xytext=(6,7),textcoords='offset points')
        ax.set_title(title,pad=15);ax.set_yticks(range(3),names);ax.set_ylim(2.5,-.5);ax.set_xlim(*bounds);ax.grid(axis='x',alpha=.18);ax.set_xlabel('Score (0–1)' if 'auroc' in key else 'Normal denominator: 3,576' if key=='test_fp' else 'GT denominator: 26')
    fig.suptitle('Experiment 25 · Shared full-normal request CDFs',x=.02,ha='left',fontsize=16,fontweight='bold');fig.subplots_adjust(left=.16,right=.98,top=.78,bottom=.23,wspace=.3)
    fig.text(.02,.06,'R04 development: 19 test videos / 8,154 frames. Same PCA banks, process and request CDFs; each gate fits its own normal q99.\nTau = 56 source frames, fixed from 139 complete normal FIT gaps. Sources: request_calibration_diagnostic.json / comparison.csv.',fontsize=10,color='#475467');fig.savefig(root/'request_calibration_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),sharex=True);x=np.arange(4);width=.24;labels=['Observed','Initial missing','Missing ≤ 56','Missing > 56']
    for ax,kind,title in zip(axes,['normal','anomaly'],['Normal false-positive frames','Anomaly true-positive frames']):
        for i,e in enumerate(AFTER):
            vals=[d['age_strata'][e][s]['alarms'][kind] for s in STATE_NAMES];bars=ax.bar(x+(i-1)*width,vals,width,color=colors[i],label=names[i]);ax.bar_label(bars,padding=13 if i==1 else 3,fontsize=9)
        ax.set_title(title);ax.set_xticks(x,labels,rotation=15,ha='right');ax.set_ylabel('Frames');ax.set_ylim(0,max(d['age_strata'][e][s]['alarms'][kind] for e in AFTER for s in STATE_NAMES)*1.2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    handles,leglabels=axes[0].get_legend_handles_labels();fig.legend(handles,leglabels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False);fig.suptitle('Fixed age subsets · same membership across all three routes',fontsize=15,fontweight='bold',x=.02,ha='left');fig.subplots_adjust(top=.72,bottom=.25,wspace=.25)
    base=d['age_strata']['25_hold'];den='; '.join(f"{labels[i]} {base[s]['normal_frames']}/{base[s]['anomaly_frames']}" for i,s in enumerate(STATE_NAMES))
    fig.text(.02,.05,'Subset denominators (normal/anomaly frames): '+den+'\nAge is evaluated at sampled frames and causally held to dense frames. Source: request_calibration_diagnostic.json.',fontsize=9,color='#475467');fig.savefig(root/'age_strata_alarms.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
