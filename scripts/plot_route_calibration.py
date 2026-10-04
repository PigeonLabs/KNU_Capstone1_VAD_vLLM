"""Visualize normal holdout failures and test observation-stratum changes."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    out=Path('results/experiment20');d=json.loads((out/'normal_holdout.json').read_text());tail=json.loads((out/'normal_tail_diagnostic.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');ids=d['holdout_sequences'];x=np.arange(len(ids))
    for offset,e,color in [(-.18,'19','#718096'),(.18,'20','#137c8b')]:
        values=[next(r['held_out_frame_alarms'] for r in d['folds'] if r['variant']==e and r['held_out_sequence']==s) for s in ids]
        bars=axes[0].bar(x+offset,values,.36,label=f'Experiment {e}',color=color);axes[0].bar_label(bars,padding=3)
    axes[0].set(xticks=x,xticklabels=ids,ylim=(0,41),ylabel='False-alarm source frames',xlabel='Held-out NORMAL sequence',title='Normal holdout alarms: 44 -> 68 / 1,920');axes[0].legend(frameon=False)
    rows=tail['added_alarms'];positions=np.arange(len(rows));route=[];role=[]
    for row in rows:
        r=row['new_visual_exceedances'][0];route.append(r['raw_residual']/r['route_reference_max']);role.append(r['raw_residual']/r['role_reference_max'])
    axes[1].scatter(positions,route,label='Residual / route maximum',s=50,color='#c45d40');axes[1].scatter(positions,role,label='Residual / role-wide maximum',s=50,color='#137c8b');axes[1].axhline(1,color='gray',ls='--',lw=1)
    axes[1].set(xticks=positions,xticklabels=[f'{r["sequence"]}/{r["source_frame_index"]}' for r in rows],ylim=(0,1.65),ylabel='Ratio to normal reference maximum',xlabel='Extra alarm: sequence / source frame',title='All six extra samples cross only the route tail');axes[1].legend(frameon=False,loc='upper right',fontsize=9)
    fig.savefig(out/'route_holdout.png',dpi=170);plt.close(fig)
    diag=json.loads((out/'route_diagnostic.json').read_text())['strata_relation_observed'];fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained');x=np.arange(2)
    for ax,label,denom,title in zip(axes,['normal','anomaly'],['normal_frames','anomaly_frames'],['Normal false-positive rate','Anomaly frame recall']):
        for offset,e,key,color in [(-.18,'19','alarms_before','#718096'),(.18,'20','alarms_after','#137c8b')]:
            bars=ax.bar(x+offset,[100*diag[k][key][label]/diag[k][denom] for k in ['False','True']],.36,label=f'Experiment {e}',color=color);ax.bar_label(bars,fmt='%.2f%%',padding=3)
        ax.set(xticks=x,xticklabels=[f'Unobserved\n(n={diag["False"][denom]:,})',f'Observed\n(n={diag["True"][denom]:,})'],ylim=(0,23),ylabel='Percent of indicated subset',title=title);ax.legend(frameon=False,loc='upper right')
    fig.suptitle('R04 development: fixed masks and PCA, each model uses its normal q99',fontsize=12);fig.savefig(out/'observation_strata.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
