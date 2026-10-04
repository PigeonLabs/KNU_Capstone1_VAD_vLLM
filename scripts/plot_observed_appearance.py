"""Plot rates on fixed observed/unobserved frame subsets for experiment19."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    out=Path('results/experiment19');d=json.loads((out/'appearance_diagnostic.json').read_text());strata=d['strata_relation_observed']
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    x=np.arange(2)
    for ax,label,denom,title in zip(axes,['normal','anomaly'],['normal_frames','anomaly_frames'],['Normal false-positive rate','Anomaly frame recall']):
        for offset,exp,key,color in [(-.18,'18','alarms_before','#718096'),(.18,'19','alarms_after','#137c8b')]:
            values=[100*strata[k][key][label]/strata[k][denom] for k in ['False','True']]
            bars=ax.bar(x+offset,values,.36,label=f'Experiment {exp}',color=color);ax.bar_label(bars,fmt='%.2f%%',padding=3)
        ax.set_xticks(x,[f'Unobserved\n(n={strata["False"][denom]:,})',f'Observed\n(n={strata["True"][denom]:,})']);ax.set_ylabel('Percent of the indicated subset');ax.set_title(title);ax.set_ylim(0,30);ax.legend(frameon=False,loc='upper right')
    fig.suptitle('R04: same relation masks, same normal q99; appearance routing changed',fontsize=12)
    fig.savefig(out/'observation_strata.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
