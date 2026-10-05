"""Render source-backed tables and figures for the completed association pilot."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ipad_vad.learned_detector import write
OUT=Path('results/experiment42')
def main():
    result=json.loads((OUT/'metrics.json').read_text());old=json.loads(Path('results/experiment41/metrics.json').read_text());history=json.loads((OUT/'history.json').read_text());summary=[]
    for arm in ['A','B','C','D']:
        for branch in ['41 IoU','42 raw','42 learned']:
            source=old['scene_macro'] if branch=='41 IoU' else [r for r in result['scene_macro'] if r['branch']==branch.split()[1]];rows=[r for r in source if r['run']==arm or r['run'].startswith(arm+'_')];keys=[k for k in rows[0] if k not in ['run','branch']];summary.append({'arm':arm,'branch':branch,'n_visual_seeds':len(rows),'metrics':{k:{'mean':float(np.mean([r[k] for r in rows])),'std':float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else None} for k in keys}})
    def table(rows):
        lines=['| branch | arm | Visual AUC | Combined AUC | Combined AP | FPR% | recall% | event% |','|---|---|---|---|---|---|---|---|']
        for r in rows:
            values=[]
            for k in ['visual_auroc','combined_auroc','combined_average_precision','normal_fpr','anomaly_recall','event_coverage']:
                scale=100 if k in ['normal_fpr','anomaly_recall','event_coverage'] else 1;digits=2 if scale==100 else 4;m=r['metrics'][k];s=f'{m["mean"]*scale:.{digits}f}';s+=f' ± {m["std"]*scale:.{digits}f}' if m['std'] is not None else '';values.append(s)
            lines.append('| '+' | '.join([r['branch'],r['arm'],*values])+' |')
        return '\n'.join(lines)
    tables=['# 실험42 결과 표','',table(summary),'','C/D는 기존 visual3seed 평균±표본 SD. Association seed는42하나. 공정별 지표의 동일가중 macro이며 gate/q99는정상자료로고정. 41과42의q99는재적합후정확히같다.','', '## 공정·run별 결과','', '| branch | scene | run | Visual AUC | Combined AUC | AP | FPR% | recall% | event detected/total |','|---|---|---|---|---|---|---|---|---|']
    for r in result['variants']:
        tables.append(f'| {r["branch"]} | {r["scene"]} | {r["run"]} | {r["metrics"]["visual"]["auroc"]:.6f} | {r["metrics"]["combined"]["auroc"]:.6f} | {r["metrics"]["combined"]["average_precision"]:.6f} | {r["normal_fpr"]*100:.3f} | {r["anomaly_recall"]*100:.3f} | {r["events"]["detected_events"]}/{r["events"]["events"]} |')
    (OUT/'report_tables.md').write_text('\n'.join(tables)+'\n');write(OUT/'report_summary.json',{'groups':summary});(OUT/'summary_table.md').write_text(table([r for r in summary if r['branch']!='42 raw'])+'\n');figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained');epochs=[r['epoch'] for r in history]
    for partition,color in [('train','#2563eb'),('validation','#ea580c')]:
        axs[0].plot(epochs,[r[partition]['loss'] for r in history],label=partition,color=color,marker='o',ms=3);axs[1].plot(epochs,[r[partition]['negative'] for r in history],label=partition,color=color,marker='o',ms=3)
    for ax,title in zip(axs,['Balanced weak-supervision objective','Negative-pair hinge loss']):ax.set(xlabel='Epoch (0 = identity adapter)',ylabel='Loss',title=title);ax.legend();ax.grid(alpha=.2);ax.set_xticks([0,5,10,15,20])
    fig.suptitle('Experiment 42 | 8,192-parameter shared metric adapter\nValidation has only 5 reviewed negatives from 4 R04 videos',fontsize=13);fig.savefig(figdir/'training.png',dpi=180);plt.close(fig)
    diagnostic=json.loads((OUT/'association_diagnostics.json').read_text())['counts'];fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,split in zip(axs,['training','testing']):
        rr=[r for r in diagnostic if r['split']==split];x=np.arange(3)
        for i,r in enumerate(rr):
            vals=[r.get(k,0) for k in ['unmatched_high_confidence_pairs','geometry_eligible_pairs','accepted_appearance_edges']];bars=ax.bar(x+(i-.5)*.32,vals,.32,label=r['branch'],color=['#64748b','#0d9488'][i]);ax.bar_label(bars,padding=3)
        ax.set_xticks(x,['Unmatched pairs','Geometry eligible','Accepted links']);ax.set_ylim(0,14);ax.set_ylabel('Count');ax.set_title('Normal 111 videos' if split=='training' else 'Test 66 videos');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('R04 material only | Other roles use the original IoU tracker',fontsize=13);fig.savefig(figdir/'association_funnel.png',dpi=180);plt.close(fig)
    print(table([r for r in summary if r['branch']!='42 raw']))
if __name__=='__main__':main()
