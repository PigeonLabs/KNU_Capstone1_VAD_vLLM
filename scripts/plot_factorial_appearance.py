"""Show all prespecified appearance-conditioning cells without winner selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('results/experiment21');d=json.loads((root/'factorial_diagnostic.json').read_text());h=json.loads((root/'normal_holdout.json').read_text());grid=[['18','21_infer'],['21_fit','19']];events={}
    for a,b in [('18','21_fit'),('18','21_infer'),('21_fit','19')]:events.update(json.loads(Path(f'results/comparison{a}_{b}/events.json').read_text())['summaries'])
    definitions=[('Normal holdout false-alarm frames',lambda e:h['totals'][e]['held_out_frame_alarms'],'d'),('Test combined AUROC',lambda e:d['cells'][e]['metrics']['combined']['auroc'],'.4f'),('Test normal false-alarm frames',lambda e:d['cells'][e]['alarms']['normal'],'d'),('Test anomaly detected frames',lambda e:d['cells'][e]['alarms']['anomaly'],'d'),('Detected GT intervals / 26',lambda e:events[e]['detected_events'],'d'),('Test combined average precision',lambda e:d['cells'][e]['metrics']['combined']['average_precision'],'.4f')]
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
    for ax,(title,value,fmt) in zip(axes.flat,definitions):
        ax.imshow(np.zeros((2,2)),cmap='Greys',vmin=0,vmax=1)
        for a in range(2):
            for b in range(2):
                e=grid[a][b];ax.text(b,a,f'{value(e):{fmt}}\n({e})',ha='center',va='center',fontsize=13,color='#15566a')
        ax.set(xticks=[0,1],xticklabels=['B off','B on'],yticks=[0,1],yticklabels=['A off','A on'],title=title)
        ax.set_xticks([.5],minor=True);ax.set_yticks([.5],minor=True);ax.grid(which='minor',color='#d0d5dd');ax.tick_params(which='minor',bottom=False,left=False)
    fig.suptitle('R04 development: A = observed FIT restriction; B = missing-relation pooled inference',fontsize=13);fig.savefig(root/'factorial_matrix.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');variants=['18','21_fit','21_infer','19'];x=np.arange(5);colors=['#718096','#137c8b','#c38338','#815ab1']
    for i,(e,color) in enumerate(zip(variants,colors)):
        counts=[next(r['held_out_frame_alarms'] for r in h['folds'] if r['variant']==e and r['held_out_sequence']==s) for s in h['holdout_sequences']];axes[0].bar(x+(i-1.5)*.18,counts,.18,label=e,color=color)
    axes[0].set(xticks=x,xticklabels=h['holdout_sequences'],ylabel='False-alarm source frames',xlabel='Held-out normal sequence',title='Normal holdout differs by video',ylim=(0,36));axes[0].legend(frameon=False,ncol=2)
    for i,e in enumerate(variants):
        raw=d['strata_relation_observed'][e];normal=[raw[k]['alarms']['normal'] for k in ['False','True']];anomaly=[raw[k]['alarms']['anomaly'] for k in ['False','True']]
        axes[1].text(.03,.86-i*.21,f'{e:>8}: missing {normal[0]:3}/{anomaly[0]:3} | observed {normal[1]:3}/{anomaly[1]:3}',fontfamily='monospace',fontsize=11,transform=axes[1].transAxes)
    axes[1].axis('off');axes[1].set_title('Test alarm frames: normal / anomaly');fig.savefig(root/'normal_and_observed.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
