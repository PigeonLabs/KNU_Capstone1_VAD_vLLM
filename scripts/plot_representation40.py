"""Publication-style figures sourced only from completed experiment40 records."""
import json,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_representation40 import OUT

COLORS={'A':'#626c76','B':'#287aa3','C':'#c47a16','D':'#7960a9'}
NAMES={'A':'A: frozen CLIP','B':'B: frozen MobileCLIP2','C':'C: LoRA','D':'D: full FT'}

def training_figure(dest):
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for arm in ['C','D']:
        for seed in [42,43,44]:
            run=f'{arm}_s{seed}';history=json.loads((OUT/f'{run}_history.json').read_text());training=json.loads((OUT/f'{run}_training.json').read_text());epochs=[r['epoch'] for r in history]
            style={42:'-',43:'--',44:':'}[seed]
            axes[0].plot(epochs,[r['validation']['loss'] for r in history],style,color=COLORS[arm],label=f'{arm} seed {seed}')
            axes[1].plot(epochs,[r['validation']['teacher_cosine_loss'] for r in history],style,color=COLORS[arm])
            axes[2].plot(epochs,[r['validation']['effective_rank'] for r in history],style,color=COLORS[arm])
    for ax,title,ylabel in zip(axes,['Normal validation objective','Drift from frozen teacher','Representation spread'],['Loss (lower is better for this objective)','Mean cosine distance','Effective rank (512 fixed views)']):
        ax.set(title=title,xlabel='Epoch',ylabel=ylabel);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,ncol=2);fig.suptitle('Training diagnostics | No anomaly labels used for checkpoint selection')
    fig.text(.5,.015,'Same fixed normal validation views at every epoch. Lower objective does not imply better anomaly detection. Source: *_history.json',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.95));fig.savefig(dest/'training_diagnostics.png');plt.close(fig)

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--training-only',action='store_true');args=parser.parse_args()
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':180})
    if args.training_only:
        training_figure(dest);return
    result=json.loads((OUT/'metrics.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for ax,scene in zip(axes.flat,['R01','R02','R03','R04']):
        for i,arm in enumerate(['A','B','C','D']):
            rows=[r for r in result['variants'] if r['scene']==scene and r['run'].split('_')[0]==arm];y=[r['metrics']['combined']['auroc'] for r in rows]
            ax.scatter(i,np.mean(y),marker='D',s=55,color=COLORS[arm],zorder=3)
            if len(y)>1:ax.scatter(np.arange(len(y))*.08+i-.08,y,color=COLORS[arm],marker='o',s=20,alpha=.7);ax.errorbar(i,np.mean(y),yerr=np.std(y,ddof=1),color=COLORS[arm],capsize=5)
            ax.annotate(f'{np.mean(y):.3f}',(i,np.mean(y)),xytext=(0,13),textcoords='offset points',ha='center',fontsize=9)
        ax.set_title(scene);ax.set_xticks(range(4),['A','B','C','D']);ax.set_ylim(0,1.06);ax.set_ylabel('Combined AUROC');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Experiment 40 | Shared visual representation, separate process models',fontsize=14)
    fig.text(.5,.015,'A/B: one frozen run. C/D: dots = seeds 42/43/44; diamond = mean; whiskers = sample SD (not confidence intervals).\nSame test frames/boxes/phases within each process. Source: results/experiment40/metrics.json',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.96));fig.savefig(dest/'process_comparison.png');plt.close(fig)
    training_figure(dest)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric,title in zip(axes,['normal_fpr','anomaly_recall'],['Normal frame false-positive rate','Anomaly frame recall']):
        for j,arm in enumerate(['A','B','C','D']):
            for i,scene in enumerate(['R01','R02','R03','R04']):
                y=[r[metric]*100 for r in result['variants'] if r['scene']==scene and r['run'].split('_')[0]==arm];x=i+(j-1.5)*.17
                ax.scatter(x,np.mean(y),color=COLORS[arm],marker=['s','^','o','D'][j],label=NAMES[arm] if i==0 else None)
                if len(y)>1:ax.errorbar(x,np.mean(y),yerr=np.std(y,ddof=1),color=COLORS[arm],capsize=3)
        ax.set(title=title,ylabel='Frames (%)',xticks=range(4),xticklabels=['R01','R02','R03','R04'],ylim=(0,20 if metric=='normal_fpr' else 50));ax.grid(axis='y',alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Operating points | Separate normal q99 per run and process')
    fig.text(.5,.015,'Strict score > q99. Axes: FPR 0–20%, recall 0–50%.\nWhiskers = 3-seed sample SD, not confidence intervals. Source: metrics.json',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(dest/'operating_points.png');plt.close(fig)

if __name__=='__main__':main()
