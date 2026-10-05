"""Static research figures from completed, inspectable experiment41 records."""
import argparse,json,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/experiment41');DEST=OUT/'figures'
COLORS={'A':'#626c76','B':'#287aa3','C':'#c47a16','D':'#7960a9'}

def read(path):return json.loads(Path(path).read_text())

def training():
    record=read(OUT/'training.json');h=record['history'];fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for scene,color,style in zip(['R01','R02','R03','R04'],COLORS.values(),['-','--',':','-.']):
        axes[0].plot([r['epoch'] for r in h],[r['scene_loss'][scene] for r in h],style,color=color,label=scene,marker='o',markersize=3)
    axes[0].set(title='Normal validation loss by process',xlabel='Epoch',ylabel='Final decoder objective',ylim=(0,None));axes[0].legend(ncol=2)
    axes[1].plot([r['epoch'] for r in h],[r['macro_scene_loss'] for r in h],'-o',color='#287aa3',label='Validation: process macro')
    axes[1].plot([r['epoch'] for r in h[1:]],[r['train_loss_mean_batches'] for r in h[1:]],'--s',color='#c47a16',label='Training: mean batch loss')
    axes[1].set(title='Checkpoint selection and training',xlabel='Epoch',ylabel='Final decoder objective',ylim=(0,None));axes[1].legend(fontsize=9)
    for ax in axes:ax.set_xticks(range(6));ax.grid(alpha=.2)
    fig.suptitle('Experiment 41 | Shared GroundingDINO decoder partial fine-tuning')
    fig.text(.5,.018,'Normal weak boxes: 178 train / 50 validation frames. Validation selected epoch 5.\nLoss measures agreement with approximate supervision, not independent detector accuracy. Source: training.json',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(DEST/'training.png');plt.close(fig)

def performance():
    records={40:read('results/experiment40/metrics.json'),41:read(OUT/'metrics.json')}
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharey=True)
    for ax,scene in zip(axes.flat,['R01','R02','R03','R04']):
        for j,arm in enumerate(COLORS):
            means=[]
            for exp,shift,marker in [(40,-.12,'o'),(41,.12,'D')]:
                ys=[r['metrics']['combined']['auroc'] for r in records[exp]['variants'] if r['scene']==scene and r['run'].split('_')[0]==arm];mu=np.mean(ys);means.append(mu)
                ax.scatter(j+shift,mu,color=COLORS[arm],facecolors='none' if exp==40 else COLORS[arm],marker=marker,s=60,zorder=3,label=f'{exp}: '+('frozen detector' if exp==40 else 'learned detector') if j==0 else None)
                if len(ys)>1:ax.errorbar(j+shift,mu,yerr=np.std(ys,ddof=1),color=COLORS[arm],capsize=3)
            ax.plot([j-.12,j+.12],means,color=COLORS[arm],alpha=.6)
        ax.set(title=scene,xticks=range(4),xticklabels=['A: CLIP','B: Mobile','C: LoRA','D: full FT'],ylabel='Combined AUROC',ylim=(0,1));ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Frozen versus learned detector | Same eight visual encoders')
    fig.text(.5,.014,'Circle = frozen detector; diamond = learned detector. C/D whiskers = prior visual seed SD, not confidence intervals.\nOne detector seed; added weak boxes. R04 in 41: unsupported dwell is explicitly unavailable. Source: experiment40/41 metrics.json',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.95));fig.savefig(DEST/'paired_auroc.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,metric,title in zip(axes,['normal_fpr','anomaly_recall'],['Normal frame false-positive rate','Anomaly frame recall']):
        for j,arm in enumerate(COLORS):
            vals=[]
            for exp,shift,marker in [(40,-.12,'o'),(41,.12,'D')]:
                ys=[r[metric]*100 for r in records[exp]['scene_macro'] if r['run'].split('_')[0]==arm];mu=np.mean(ys);vals.append(mu)
                ax.scatter(j+shift,mu,color=COLORS[arm],facecolors='none' if exp==40 else COLORS[arm],marker=marker,s=60)
                if len(ys)>1:ax.errorbar(j+shift,mu,yerr=np.std(ys,ddof=1),color=COLORS[arm],capsize=3)
            ax.plot([j-.12,j+.12],vals,color=COLORS[arm],alpha=.6)
        ax.set(title=title,ylabel='Process macro (%)',xticks=range(4),xticklabels=list(COLORS),ylim=(0,None));ax.grid(axis='y',alpha=.2)
    fig.suptitle('Operating points | Each run and process uses its own normal q99')
    fig.text(.5,.018,'Circle = experiment 40; diamond = experiment 41. C/D whiskers = 3 prior visual seed sample SD.\nNormal references refitted. R04 in 41: unsupported dwell is unavailable. Source: metrics.json',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.95));fig.savefig(DEST/'operating_points.png');plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training-only',action='store_true');args=parser.parse_args();DEST.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':180})
    training()
    if not args.training_only:performance()

if __name__=='__main__':main()
