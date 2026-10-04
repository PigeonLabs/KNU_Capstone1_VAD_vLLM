"""Diagnose a bounded score whose normal quantile makes strict alarms impossible."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);args=p.parse_args()
    root=Path(f'artifacts/experiment{args.experiment}');out=Path(f'results/experiment{args.experiment}')
    m=json.loads((out/'metrics.json').read_text());q=m['normal_q99_threshold']
    with np.load(root/'normal_calibration_scores.npz',allow_pickle=False) as d:
        n=len(d['combined']);rows=[]
        for seq in np.unique(d['sequence']):
            use=(d['sequence']==seq)&(d['combined']==1)
            rows.append({'sequence':str(seq),'samples':int(np.sum(d['sequence']==seq)),'combined_at_one':int(use.sum()),'dwell_ages_at_ceiling':d['dwell_age'][use].tolist(),'phases_at_ceiling':d['phases'][use].tolist(),'entry_contexts_at_ceiling':d['dwell_entry_context'][use].tolist()})
        branches={k:{'at_one':int(np.sum(d[k]==1)),'maximum':float(d[k].max())} for k in ['visual','transition','dwell','combined']}
        assert np.quantile(d['combined'],.99,method='higher')==q
        sorted_scores=np.sort(d['combined'])
    counts={'normal':0,'anomaly':0};maximum=0.;alarms=0
    for path in (root/'predictions').glob('*.npz'):
        with np.load(path,allow_pickle=False) as d:
            maximum=max(maximum,float(d['combined'].max()));alarms+=int(np.sum(d['combined']>q))
            for value,name in [(0,'normal'),(1,'anomaly')]:counts[name]+=int(np.sum((d['labels']==value)&(d['combined']==1)))
    result={'calibration_samples':n,'quantile_higher_sorted_zero_based_index':int(np.ceil(.99*(n-1))),'normal_q99':q,'calibration_branches':branches,'calibration_at_one_fraction':branches['combined']['at_one']/n,'calibration_by_video':rows,'test_score_max':maximum,'test_at_one':counts,'strict_threshold_test_alarms':alarms,'declared_score_upper_bound':1.,'alarm_impossible_for_bounded_scores':q>=1.,'interpretation':'Empirical complete-run percentile saturates beyond FIT maximum. A normal q99 equal to its closed upper bound makes strict > alarms impossible, regardless of ranking. No threshold comparison operator or tie rule was changed.'}
    if q<1:
        result['interpretation']='Normal q99 is below the closed score upper bound, so strict alarms are not structurally impossible. This does not establish useful incremental detection. No threshold comparison operator or tie rule was changed.'
    assert maximum<=1
    if q>=1:assert alarms==0
    (out/'threshold_ceiling.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');start=max(0,n-20);x=np.arange(start,n)
    ax.plot(x,sorted_scores[start:],'o-',label='Top 20 normal calibration scores')
    ax.axhline(q,color='crimson',linestyle='--',label=f'Normal q99 = {q:.6f}')
    at_one=sorted_scores[start:]==1;ax.scatter(x[at_one],sorted_scores[start:][at_one],color='crimson',zorder=3)
    ax.set(xlabel='Sorted calibration index (zero-based)',ylabel='Combined score',xticks=np.unique(np.r_[np.arange(start,n,5),n-1]),ylim=(sorted_scores[start]-.001,1.0015),title=f'Exp {args.experiment}: {branches["combined"]["at_one"]} / {n} normal scores at the closed upper bound')
    ax.legend(loc='lower right');fig.savefig(out/'threshold_ceiling.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
