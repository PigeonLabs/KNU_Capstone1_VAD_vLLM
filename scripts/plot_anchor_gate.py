"""Plot observed coverage and six qualitative normal cases; summarize alarm branches."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('results/experiment17');normal=json.loads((out/'normal_anchor_audit.json').read_text());diag=json.loads((out/'gate_diagnostic.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');x=np.arange(3)
    for shift,key,label in [(-.18,'relations_before','Exp 16'),(.18,'relations_after','Exp 17 gate')]:
        values=[100*diag['groups'][g][key]/diag['groups'][g]['samples'] for g in ['fit','calibration','test']]
        bars=axes[0].bar(x+shift,values,.36,label=label);axes[0].bar_label(bars,fmt='%.1f%%',padding=3)
    axes[0].set(xticks=x,xticklabels=['Normal FIT','Normal calibration','Test'],ylim=(0,100),ylabel='Sampled relation availability (%)',title='Availability is not object accuracy');axes[0].legend(loc='upper center',ncol=2)
    cases=normal['qualitative_cases'];assert all(len(r['candidates'])==1 for r in cases)
    values=[r['candidates'][0]['margin'] for r in cases];names=[f'{r["sequence"]}/{r["source_frame"]} '+('blade' if i==5 else 'vise') for i,r in enumerate(cases)]
    axes[1].barh(names,values,color=['tab:blue' if i==5 else 'tab:orange' for i in range(6)]);axes[1].axvline(0,color='black',linestyle='--');axes[1].invert_yaxis();axes[1].set(xlabel='Cosine margin: blade minus vise',title='Six normal qualitative cases (not a test set)')
    fig.savefig(out/'anchor_gate.png',dpi=160);plt.close(fig)
    m=json.loads((out/'metrics.json').read_text());q=m['normal_q99_threshold'];tot={k:{'normal':0,'anomaly':0} for k in ['visual','transition','dwell']};maxd=maxage=0.
    for path in Path('artifacts/experiment17/predictions').glob('*.npz'):
        with np.load(path) as d:
            for branch in tot:
                for value,name in [(0,'normal'),(1,'anomaly')]:tot[branch][name]+=int(np.sum((d[branch]>q)&(d['labels']==value)))
            if d['dwell_valid'].any():maxd=max(maxd,float(d['dwell'][d['dwell_valid']].max()));maxage=max(maxage,float(d['dwell_age'][d['dwell_valid']].max()))
            np.testing.assert_array_equal(d['combined']>q,d['visual']>q)
    r={'q99':q,'branch_exceedances':tot,'max_valid_dwell_score':maxd,'max_valid_dwell_age':maxage,'combined_alarm_equals_visual_alarm':True,'note':'Branch exceedances overlap; descriptive attribution at final q99, not separate branch-optimal thresholds.'}
    (out/'branch_alarms.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
