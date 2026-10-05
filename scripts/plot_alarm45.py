"""Layout-only refinement after visual inspection; read unchanged report data."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from candidates45_common import OUT,config,sha,write


def main():
    data=json.loads((OUT/'report_data.json').read_text());rows=data['process']
    groups=['B','C','clip_L','siglip2_L','dinov2_L'];colors=['#636b74','#326b9b','#c08027','#78803c','#b8668b'];markers=['o','s','^','D','X']
    selected=[r for r in rows if r['group'] in groups]
    xmax=max(100*(r['metrics']['normal_fpr']['mean']+(r['metrics']['normal_fpr']['sample_std'] or 0)) for r in selected)
    ymax=max(100*(r['metrics']['anomaly_recall']['mean']+(r['metrics']['anomaly_recall']['sample_std'] or 0)) for r in selected)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#666666','text.color':'#20252b','axes.labelcolor':'#20252b'})
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,scene in zip(axes.flat,config()['scenes']):
        for group,color,marker in zip(groups,colors,markers):
            row=next(r for r in rows if r['group']==group and r['scene']==scene);m=row['metrics']
            ax.errorbar(100*m['normal_fpr']['mean'],100*m['anomaly_recall']['mean'],
                xerr=100*(m['normal_fpr']['sample_std'] or 0),yerr=100*(m['anomaly_recall']['sample_std'] or 0),
                fmt=marker,color=color,markersize=8,capsize=3,label=row['label'])
        ax.set_title(scene,loc='left');ax.set_xlim(0,max(1,xmax*1.12));ax.set_ylim(0,max(1,ymax*1.12));ax.grid(alpha=.16)
    tie=[]
    for group in ['B','C','dinov2_L']:
        m=next(r['metrics'] for r in rows if r['group']==group and r['scene']=='R01')
        tie.append((100*m['normal_fpr']['mean'],100*m['anomaly_recall']['mean']))
    if np.allclose(tie,tie[0],rtol=0,atol=1e-10):
        axes[0,0].annotate('Mobile frozen / LoRA / DINOv2-L\nsame alarm rates',xy=tie[0],xytext=(6,29),fontsize=9,
            arrowprops={'arrowstyle':'-','color':'#636b74','lw':.8},color='#50565d')
    fig.subplots_adjust(left=.10,right=.985,bottom=.29,top=.89,hspace=.30,wspace=.08)
    fig.suptitle('45 | Alarm trade-offs at each model’s own normal q99',x=.06,y=.97,ha='left')
    fig.supxlabel('Normal frame false-positive rate (%)',y=.215)
    fig.supylabel('Anomalous frame recall (%)',x=.025,y=.60)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.075),fontsize=10)
    fig.text(.06,.03,'Historical Mobile LoRA: mean ± 3-seed SD. Separate thresholds; no matched-FPR superiority claim.',fontsize=9,color='#50565d')
    fig.savefig(OUT/'alarm_tradeoffs.png',dpi=180);plt.close(fig)
    write(OUT/'plot_layout_refinement.json',{'reason':'Original shared xlabel overlapped footer at native figure size; label/legend positions refined and exact R01 overlap annotated.',
        'metrics_changed':False,'report_data_sha256':sha(OUT/'report_data.json'),'script_sha256':sha(__file__),
        'output_sha256':sha(OUT/'alarm_tradeoffs.png'),'original_report_source_retained':True})


if __name__=='__main__':main()
