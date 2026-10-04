"""Normal-only fold FPR chart and held-out duration support evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ipad_vad.context_dwell import complete_context_runs


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    root=Path('results/experiment13');m=json.loads((root/'metrics.json').read_text());ids=m['heldout_pool'];variants=list(m['summaries']);names=['No dwell (10)','State dwell (11)','Matched support (12s)','Entry dwell (12)']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');x=np.arange(len(ids));width=.19
    for i,(variant,label) in enumerate(zip(variants,names)):
        values=[next(r['frame_fpr']*100 for r in m['folds'] if r['variant']==variant and r['held_out']==seq) for seq in ids]
        axes[0].bar(x+(i-1.5)*width,values,width,label=label)
    axes[0].axhline(1,color='black',ls='--',lw=1,label='Calibration target 1%')
    axes[0].set(xticks=x,xticklabels=ids,xlabel='Held-out normal video',ylabel='False-positive frames (%)',title='R03 normal-video holdout');axes[0].legend(fontsize=8)
    pooled=[m['summaries'][v]['micro_frame_fpr']*100 for v in variants];macro=[m['summaries'][v]['macro_video_frame_fpr']*100 for v in variants]
    axes[1].bar(x-.18,pooled,.36,label='Frame-weighted');axes[1].bar(x+.18,macro,.36,label='Video mean')
    axes[1].axhline(1,color='black',ls='--',lw=1);axes[1].set(xticks=x,xticklabels=['10','11','12_support','12'],ylabel='False-positive rate (%)',title='Same 2,806 normal frames');axes[1].legend(fontsize=8)
    fig.savefig(root/'normal_holdout.png',dpi=150);plt.close(fig)
    alarms={v:[load(Path(f'artifacts/experiment13/variant{v}/holdout_{s}/heldout_scores.npz'))['dense_alarm'] for s in ids] for v in variants}
    assert all(np.array_equal(a,b) for v in ['12_support','12'] for a,b in zip(alarms['11'],alarms[v]))
    extra=[]
    for i,s in enumerate(ids):
        added=alarms['12'][i]&~alarms['10'][i];lost=~alarms['12'][i]&alarms['10'][i]
        extra.append({'held_out':s,'added_by_dwell':int(added.sum()),'lost_by_dwell':int(lost.sum())})
    held='22';fold=next(r for r in m['folds'] if r['variant']=='12' and r['held_out']==held);other=fold['calibration_ids'].split(',');source=Path('artifacts/experiment12/features/R03')
    fit_d=load(Path('artifacts/experiment13/variant12/holdout_22/normal_model.npz'))['dwell_context_durations_1_2']
    lengths={s:[length for key,length in complete_context_runs(load(source/f'training_{s}.npz')) if key==(1,2)] for s in other+[held]}
    pred=load(Path('artifacts/experiment13/variant12/holdout_22/heldout_scores.npz'));base=load(Path('artifacts/experiment13/variant10/holdout_22/heldout_scores.npz'))
    assert next(r['q99'] for r in m['folds'] if r['variant']=='10' and r['held_out']==held)==fold['q99']
    extra_sample=(pred['combined']>fold['q99'])&~(base['combined']>fold['q99']);ages=pred['dwell_age'][extra_sample]
    raw=-np.log((1+len(fit_d)-np.searchsorted(fit_d,ages,side='left'))/(1+len(fit_d)))
    reference=load(Path('artifacts/experiment13/variant12/holdout_22/normal_model.npz'))['dwell_reference']
    assert np.all(raw>reference.max()) and np.all(pred['dwell'][extra_sample]==1)
    out={'normal_only':True,'dwell_variants_have_identical_alarm_masks':True,'added_dwell_alarms_by_video':extra,
        'holdout22_context_1_to_2':{'fit_complete_durations':fit_d.tolist(),'calibration_complete_durations':{s:lengths[s] for s in other},'heldout_complete_duration':lengths[held],
            'additional_alarm_sample_ages':ages.tolist(),'additional_alarm_raw_min_max':[float(raw.min()),float(raw.max())],'calibration_raw_max':float(reference.max()),'additional_alarm_dwell_percentile_all_one':True,'additional_alarms_still_within_fit_max_duration':bool(np.all(ages<=fit_d.max())),
            'heldout_complete_duration_within_fit_range':bool(min(fit_d)<=lengths[held][0]<=max(fit_d))},
        'interpretation':'The held-out NORMAL run is within the completed-duration FIT range, yet its late ages exceed the three-video calibration raw-score tail and saturate the dwell percentile. This is a calibration-support counterexample, not proof that all dwell false alarms have the same cause.'}
    (root/'support_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
