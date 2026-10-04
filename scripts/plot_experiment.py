"""Render aggregate result figures; no raw dataset frames are published."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',default='01');args=p.parse_args()
    root=Path(f'results/experiment{args.experiment}');d=json.loads((root/'metrics.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    branches=list(d['metrics']);x=np.arange(len(branches))
    axes[0].bar(x-.18,[d['metrics'][k]['auroc'] for k in branches],.36,label='AUROC')
    axes[0].bar(x+.18,[d['metrics'][k]['average_precision'] for k in branches],.36,label='Average precision')
    axes[0].set(xticks=x,xticklabels=branches,ylim=(0,1),title=f'Experiment {args.experiment}: R01 test frames')
    axes[0].axhline(.5,color='gray',linestyle=':',linewidth=1);axes[0].legend()
    labels=['Fit normal','Calibration normal','Test']
    counts=np.array([d['phase_fit_counts'],d['phase_calibration_counts'],d['phase_test_counts']])
    fractions=counts/counts.sum(1,keepdims=True);bottom=np.zeros(3)
    for k in range(counts.shape[1]):
        axes[1].bar(labels,fractions[:,k],bottom=bottom,label=f'Phase {k}');bottom+=fractions[:,k]
    axes[1].set(ylim=(0,1),ylabel='Fraction of sampled frames',title='Observed phase distribution (not GT)');axes[1].legend()
    fig.savefig(root/'summary.png',dpi=150);plt.close(fig)
    pred=Path(f'artifacts/experiment{args.experiment}/predictions')
    chosen=['03','05','13'];fig,axes=plt.subplots(3,1,figsize=(10,7),layout='constrained')
    for ax,seq in zip(axes,chosen):
        f=np.load(pred/f'R01_{seq}.npz');t=np.arange(len(f['labels']))
        ax.fill_between(t,0,1,where=f['labels']==1,color='red',alpha=.12,label='GT anomaly')
        ax.plot(t,f['combined'],label='Combined score',linewidth=1.2)
        ax.axhline(d['normal_q99_threshold'],color='black',linestyle='--',label='Normal q99')
        ax.set(title=f'R01/{seq}',ylim=(0,1),xlabel='Source frame index (not seconds)');ax.legend(loc='upper right',fontsize=8)
    fig.savefig(root/'timelines.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
