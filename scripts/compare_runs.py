"""Compare completed experiments, preserving scope and label alignment checks."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiments',nargs='+',required=True);args=p.parse_args()
    data=[json.loads(Path(f'results/experiment{n}/metrics.json').read_text()) for n in args.experiments]
    first=data[0]
    for n,d in zip(args.experiments,data):
        for key in ('scene','seed','fit_sequences','calibration_sequences','test_sequences'):assert d[key]==first[key],key
        assert d['metrics']['combined']['frames']==first['metrics']['combined']['frames']
        base=Path(f'artifacts/experiment{args.experiments[0]}/predictions')
        other=Path(f'artifacts/experiment{n}/predictions')
        assert sorted(x.name for x in base.glob('*.npz'))==sorted(x.name for x in other.glob('*.npz'))
        for path in base.glob('*.npz'):
            with np.load(path) as a,np.load(other/path.name) as b:assert np.array_equal(a['labels'],b['labels']),path
    rows=[]
    for n,d in zip(args.experiments,data):
        row={'experiment':n}
        for branch in ('visual','process','combined'):
            for key in ('auroc','average_precision'):row[f'{branch}_{key}']=d['metrics'][branch][key]
        for key in ('normal_q99_threshold','test_normal_frame_alarm_rate','test_anomaly_frame_recall_at_q99'):row[key]=d[key]
        rows.append(row)
    root=Path('results/comparison'+'_'.join(args.experiments));root.mkdir(parents=True,exist_ok=True)
    with (root/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (root/'summary.json').write_text(json.dumps({'scene':first['scene'],'seed':first['seed'],'test_frames':first['metrics']['combined']['frames'],
        'rows':rows,'comparison_note':'Descriptive development scene; each model uses its own normal calibration q99. Not equal test FPR. No significance claim.'},indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,keys,names in ((axes[0],['combined_auroc','combined_average_precision'],['AUROC','Average precision']),
                          (axes[1],['test_normal_frame_alarm_rate','test_anomaly_frame_recall_at_q99'],['False positive rate','Anomaly recall'])):
        width=.8/len(rows);x=np.arange(2)
        for i,row in enumerate(rows):ax.bar(x+(i-(len(rows)-1)/2)*width,[row[k] for k in keys],width,label=f'Exp {row["experiment"]}')
        ax.set(xticks=x,xticklabels=names,ylim=(0,1));ax.legend()
    axes[0].set_title('R01 combined score ranking');axes[1].set_title('Each normal calibration q99')
    fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
