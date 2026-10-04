"""Plot normal holdout regression and compare against the no-dwell baseline."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('results/experiment14')
    normal=json.loads((out/'normal_holdout.json').read_text())
    rows=normal['folds'];x=np.arange(len(rows));fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for shift,key,label in [(-.18,'baseline_frame_fpr','Exp 12: age recalibration'),(.18,'frame_fpr','Exp 14: complete FIT runs')]:
        bars=ax.bar(x+shift,[100*r[key] for r in rows],.36,label=label)
        ax.bar_label(bars,fmt='%.2f%%',padding=3)
    ax.set(xticks=x,xticklabels=[r['held_out'] for r in rows],xlabel='Held-out normal video (development data)',ylabel='False positive frames (%)',ylim=(0,10),title='R03: same normal fold q99, 40 fewer false positive frames')
    ax.legend();fig.savefig(out/'normal_holdout.png',dpi=160);plt.close(fig)
    thresholds={n:json.loads(Path(f'results/experiment{n}/metrics.json').read_text())['normal_q99_threshold'] for n in ('10','14')}
    counts={'added':{'normal':0,'anomaly':0},'removed':{'normal':0,'anomaly':0}}
    for path in sorted(Path('artifacts/experiment10/predictions').glob('*.npz')):
        with np.load(path) as a,np.load(Path('artifacts/experiment14/predictions')/path.name) as b:
            np.testing.assert_array_equal(a['labels'],b['labels']);y=b['labels']
            before=a['combined']>thresholds['10'];after=b['combined']>thresholds['14']
            for key,mask in [('added',after&~before),('removed',before&~after)]:
                for value,name in [(0,'normal'),(1,'anomaly')]:counts[key][name]+=int((mask&(y==value)).sum())
    (out/'no_dwell_comparison.json').write_text(json.dumps({'thresholds':thresholds,**counts,'note':'Paired frame alarms on development R03; no test threshold tuning.'},indent=2)+'\n')
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
