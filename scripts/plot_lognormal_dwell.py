"""Visualize normal fitted scores and their realized development-test operating range."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import lognorm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ipad_vad.scoring import empirical_percentile


def main():
    root=Path('results/experiment16');tail=json.loads((root/'tail_operating_range.json').read_text());keys=list(tail['contexts'])
    fig,axes=plt.subplots(1,len(keys),figsize=(12,4),layout='constrained')
    with np.load('artifacts/experiment16/normal_model.npz') as model:
        for ax,key in zip(np.atleast_1d(axes),keys):
            a,b=key.split('->');mu,sigma=model[f'dwell_log_parameters_{a}_{b}'];durations=model[f'dwell_context_durations_{a}_{b}'];d=tail['contexts'][key]
            x=np.linspace(0,d['age_at_q99']*1.12,700)
            ax.plot(x,empirical_percentile(durations,x),label='Exp 15: empirical complete lengths')
            ax.plot(x,lognorm.cdf(x,s=sigma,scale=np.exp(mu)),label='Exp 16: lognormal CDF')
            ax.axhline(tail['q99'],color='crimson',linestyle='--',label='Exp 16 normal q99')
            ax.axvline(d['max_age'],color='gray',linestyle=':',label='Largest observed valid test age')
            ax.set(title=f'Entry context {key}',xlabel='Observed age (source frames)',ylabel='Dwell score',ylim=(-.02,1.05));ax.legend(fontsize=8,loc='lower right')
    fig.savefig(root/'duration_scores.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
