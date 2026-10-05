"""Source-backed phase-head report tables and publication figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ipad_vad.learned_detector import write
OUT=Path('results/experiment43')
def main():
    result=json.loads((OUT/'metrics.json').read_text());old=json.loads(Path('results/experiment41/metrics.json').read_text());history=json.loads((OUT/'history.json').read_text());training=json.loads((OUT/'training.json').read_text());summary=[]
    for arm in ['A','B','C','D']:
        for branch in ['41 control','43 teacher','43 linear','43 adapter']:
            source=old['scene_macro'] if branch=='41 control' else [r for r in result['scene_macro'] if r['branch']==branch.split()[1]];rr=[r for r in source if r['run']==arm or r['run'].startswith(arm+'_')];keys=[k for k in rr[0] if k not in ['run','branch']];summary.append({'arm':arm,'branch':branch,'n_visual_seeds':len(rr),'metrics':{k:{'mean':float(np.mean([r[k] for r in rr])),'std':float(np.std([r[k] for r in rr],ddof=1)) if len(rr)>1 else None} for k in keys}})
    lines=['| phase branch | visual arm | Visual AUC | Combined AUC | Combined AP | FPR% | recall% | event% |','|---|---|---|---|---|---|---|---|']
    for r in summary:
        values=[]
        for key in ['visual_auroc','combined_auroc','combined_average_precision','normal_fpr','anomaly_recall','event_coverage']:
            scale=100 if key in ['normal_fpr','anomaly_recall','event_coverage'] else 1;digits=2 if scale==100 else 4;m=r['metrics'][key];v=f'{m["mean"]*scale:.{digits}f}';v+=f' ± {m["std"]*scale:.{digits}f}' if m['std'] is not None else '';values.append(v)
        lines.append('| '+' | '.join([r['branch'],r['arm'],*values])+' |')
    table='\n'.join(lines);(OUT/'summary_table.md').write_text(table+'\n');write(OUT/'report_summary.json',{'groups':summary});full=['# 실험43 결과 표','',table,'','공정별 동일 가중 macro. C/D는 기존 visual3seed 평균±표본 SD; phase head는 공정별 seed42 하나. Run·공정·phase branch마다 별도 정상 q99를 사용하므로 operating point가 다르다.','', '## 공정·run별 결과','', '| phase | scene | visual | Visual AUC | Combined AUC | AP | FPR% | recall% | event detected/total |','|---|---|---|---|---|---|---|---|---|']
    for r in result['variants']:full.append(f'| {r["branch"]} | {r["scene"]} | {r["run"]} | {r["metrics"]["visual"]["auroc"]:.6f} | {r["metrics"]["combined"]["auroc"]:.6f} | {r["metrics"]["combined"]["average_precision"]:.6f} | {r["normal_fpr"]*100:.3f} | {r["anomaly_recall"]*100:.3f} | {r["events"]["detected_events"]}/{r["events"]["events"]} |')
    (OUT/'report_tables.md').write_text('\n'.join(full)+'\n');figdir=OUT/'figures';figdir.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});colors={'linear':'#64748b','adapter':'#0d9488'}
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,scene in zip(axs.flat,['R01','R02','R03','R04']):
        for branch in ['linear','adapter']:
            rr=history[f'{scene}_{branch}'];x=[r['epoch'] for r in rr];y=[r['validation']['macro_cross_entropy'] for r in rr];selected=next(r['selected_epoch'] for r in training['rows'] if r['scene']==scene and r['branch']==branch);ax.plot(x,y,color=colors[branch],label=branch);ax.scatter([selected],[y[selected]],color=colors[branch],marker='o',s=35,zorder=3)
        ax.set(title=scene,xlabel='Epoch',ylabel='Validation macro CE');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Experiment 43 | Per-process heads on shared frozen visual features\nWeak geometric targets; dots mark selected checkpoints, not action accuracy',fontsize=12);fig.savefig(figdir/'training.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(13,4),layout='constrained');bs=['41 control','43 teacher','43 linear','43 adapter'];cs=['#94a3b8','#2563eb','#d97706','#0d9488']
    process_summary=[]
    for ax,scene in zip(axs,['R01','R02','R03','R04']):
        means=[];std=[]
        for branch in bs:
            src=old['variants'] if branch=='41 control' else [r for r in result['variants'] if r['branch']==branch.split()[1]];rr=[r for r in src if r['scene']==scene and r['run'].startswith('C')];means.append(float(np.mean([r['metrics']['combined']['auroc'] for r in rr])));std.append(float(np.std([r['metrics']['combined']['auroc'] for r in rr],ddof=1)));process_summary.append({'scene':scene,'branch':branch,'visual':'C','combined_auroc':means[-1],'combined_auroc_std':std[-1],'combined_ap':float(np.mean([r['metrics']['combined']['average_precision'] for r in rr])),'normal_fpr':float(np.mean([r['normal_fpr'] for r in rr])),'anomaly_recall':float(np.mean([r['anomaly_recall'] for r in rr])),'event_coverage':float(np.mean([r['events']['event_coverage'] for r in rr]))})
        bars=ax.bar(np.arange(4),means,color=cs,yerr=std,capsize=3);ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=9);ax.set_xticks(range(4),['41','Teacher','Linear','Adapter'],rotation=30);ax.set_ylim(0,1);ax.set_title(scene);ax.set_ylabel('Combined AUROC');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Visual C (three historical LoRA seeds) | Phase refit and learned heads separated\nError bars: visual-seed sample SD, not confidence intervals',fontsize=12);fig.savefig(figdir/'process_comparison.png',dpi=180);plt.close(fig);write(OUT/'process_C_summary.json',process_summary);print(table)
if __name__=='__main__':main()
