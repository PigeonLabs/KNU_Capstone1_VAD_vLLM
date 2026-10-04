"""Publication-safe aggregate normal transition coverage; no source image pixels."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('results/experiment29');audit=json.loads((out/'coverage_audit.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,5.5));plt.rcParams.update({'font.size':11})
    for ax,part,title in zip(axes,['fit','calibration'],['Normal FIT · 20 videos','Normal calibration · 5 videos']):
        c=np.array(audit['coverage'][part]['observed']['counts']);v=np.array(audit['coverage'][part]['observed']['distinct_videos']);im=ax.imshow(v,cmap='Blues',vmin=0,vmax=20)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{c[i,j]} / {v[i,j]}',ha='center',va='center',color='white' if v[i,j]>10 else '#17212b',fontsize=12)
        ax.set_xticks(range(4));ax.set_yticks(range(4));ax.set_xlabel('Current latent phase');ax.set_ylabel('Previous latent phase');ax.set_title(title,pad=12);ax.add_patch(plt.Rectangle((1.5,2.5),1,1,fill=False,edgecolor='#D14936',lw=2.5))
    fig.suptitle('Observed transition support · samples / distinct videos',fontsize=15,fontweight='bold',x=.02,ha='left');fig.subplots_adjust(top=.80,bottom=.23,left=.07,right=.87,wspace=.32);cax=fig.add_axes([.90,.27,.02,.46]);fig.colorbar(im,cax=cax,label='Distinct videos (shared scale)');fig.text(.02,.05,'R04 normal-only audit. Red outline: 3→2 (FIT 4 samples / 4 videos; calibration 2 / 1).\nLatent IDs are not semantic action labels. Sample counts are not independent observations.\nSource: coverage_audit.json. No test performance was evaluated.',fontsize=10,color='#475467');fig.savefig(out/'observed_transition_coverage.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
