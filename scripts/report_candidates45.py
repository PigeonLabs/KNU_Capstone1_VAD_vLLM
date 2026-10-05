"""Source-backed tables and static research figures; no unexecuted results."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from candidates45_common import OUT,config,sha,write

GROUPS={'A':['A'],'B':['B'],'C':[f'C_s{s}' for s in [42,43,44]],
        **{k:[k] for k in config()['candidates']}}
LABELS={'A':'CLIP-B frozen','B':'MobileCLIP2-S2 frozen','C':'MobileCLIP2-S2 LoRA',
        'clip_L':'CLIP-L frozen','siglip2_L':'SigLIP2-L frozen','dinov2_L':'DINOv2-L frozen'}
KEYS=['visual_auroc','combined_auroc','combined_average_precision','normal_fpr','anomaly_recall','event_coverage']


def aggregate(rows):
    return {k:{'mean':float(np.mean([r[k] for r in rows])),
               'sample_std':float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else None} for k in KEYS}


def number(stats,percent=False):
    factor=100 if percent else 1;digits=2 if percent else 4
    text=f'{stats["mean"]*factor:.{digits}f}'
    if stats['sample_std'] is not None:text+=f' ± {stats["sample_std"]*factor:.{digits}f}'
    return text+('%' if percent else '')


def main():
    metrics=json.loads((OUT/'metrics.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
    validation=json.loads((OUT/'validation.json').read_text());assert validation['protocols_verified']
    summary=[];process=[]
    for group,runs in GROUPS.items():
        rows=[r for r in metrics['scene_macro'] if r['run'] in runs];assert len(rows)==len(runs)
        summary.append({'group':group,'label':LABELS[group],'runs':runs,'metrics':aggregate(rows)})
        for scene in config()['scenes']:
            rows=[r for r in metrics['variants'] if r['run'] in runs and r['scene']==scene]
            flat=[{'visual_auroc':r['metrics']['visual']['auroc'],'combined_auroc':r['metrics']['combined']['auroc'],
                   'combined_average_precision':r['metrics']['combined']['average_precision'],
                   **{k:r[k] for k in ['normal_fpr','anomaly_recall']},'event_coverage':r['events']['event_coverage']} for r in rows]
            process.append({'group':group,'scene':scene,'label':LABELS[group],'metrics':aggregate(flat),
                'normal_holdout_fp_mean':float(np.mean([r['normal_holdout_fp'] for r in rows])),
                'normal_holdout_frames':rows[0]['normal_holdout_frames'],
                'normal_eligible_all':all(r['normal_eligible'] for r in rows)})
    table=['| Encoder | Visual AUROC | Combined AUROC | Combined AP | 정상 FPR | 이상 recall | Event coverage |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        selected=' **(normal proxy 선정)**' if row['group']==selection['selected'] else ''
        table.append('| '+row['label']+selected+' | '+' | '.join(number(row['metrics'][k],i>=3) for i,k in enumerate(KEYS))+' |')
    (OUT/'summary_table.md').write_text('\n'.join(table)+'\n')
    table=['| 공정 | Encoder | Visual AUROC | Combined AUROC | Combined AP | 정상 FPR | 이상 recall | Event coverage |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for scene in config()['scenes']:
        for row in [r for r in process if r['scene']==scene]:
            table.append('| '+scene+' | '+row['label']+' | '+' | '.join(number(row['metrics'][k],i>=3) for i,k in enumerate(KEYS))+' |')
    (OUT/'process_table.md').write_text('\n'.join(table)+'\n')
    models=[]
    for name in config()['candidates']:
        smoke=json.loads((OUT/f'{name}_smoke.json').read_text());probe=json.loads((OUT/f'{name}_probe.json').read_text())
        normal=json.loads((OUT/f'{name}_normal_extraction.json').read_text());test=json.loads((OUT/f'{name}_test_extraction.json').read_text())
        models.append({'candidate':name,'visual_parameters':smoke['parameters'],'dimension':smoke['feature_dimension'],
            'native_input':smoke['input_shape'][2:],'proxy_auroc':probe['selection_score'],
            'proxy_encoder_ms_per_view':1000*probe['encoder_seconds']/probe['encoded_views'],
            'normal_views':normal['views'],'test_views':test['views'],
            'normal_extraction_seconds':normal['elapsed_seconds'],'test_extraction_seconds':test['elapsed_seconds'],
            'peak_allocated_GiB':max(normal['gpu_peak_allocated_bytes'],test['gpu_peak_allocated_bytes'])/2**30})
    normal_audit=json.loads((OUT/'normal_models_audit.json').read_text());ranks=[]
    for row in normal_audit['full']:
        spaces=row['subspaces'];ranks.append({'run':row['run'],'scene':row['scene'],'spaces':len(spaces),
            'rank_min':min(s['rank'] for s in spaces),'rank_max':max(s['rank'] for s in spaces),
            'rank_mean':float(np.mean([s['rank'] for s in spaces])),
            'at_rank_cap':sum(s['rank']==32 for s in spaces),'q99':row['q99']})
    sources=[OUT/'metrics.json',OUT/'selection.json',OUT/'validation.json',OUT/'normal_models_audit.json']
    sources += [OUT/f'{n}_{suffix}.json' for n in config()['candidates'] for suffix in ['smoke','probe','normal_extraction','test_extraction']]
    write(OUT/'report_data.json',{'summary':summary,'process':process,'models':models,'ranks':ranks,
        'source_sha256':{str(p):sha(p) for p in sources},
        'notes':['Macro across four process-specific evaluations; development data, not independent confirmation.',
                 'C error bars are sample SD across three historical visual seeds, not a confidence interval.',
                 'Each model/process uses its own normal q99; FPR and recall comparisons are not at matched FPR.',
                 'Proxy is synthetic local appearance, not actual defects; heldout video selection is fixed before test.']})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#666666','text.color':'#20252b','axes.labelcolor':'#20252b'})
    fig,ax=plt.subplots(figsize=(12,6.6));y=np.arange(len(summary));height=.33
    for j,(key,color,label,hatch) in enumerate([('visual_auroc','#326b9b','Visual AUROC',''),('combined_auroc','#c08027','Combined AUROC','//')]):
        vals=[r['metrics'][key]['mean'] for r in summary];err=[r['metrics'][key]['sample_std'] or 0 for r in summary]
        bars=ax.barh(y+(j-.5)*height,vals,height=height,color=color,label=label,hatch=hatch,
                     xerr=err,capsize=3,error_kw={'ecolor':'#20252b','elinewidth':1})
        for bar,value in zip(bars,vals):ax.text(value+.012,bar.get_y()+bar.get_height()/2,f'{value:.4f}',va='center',fontsize=10)
    ax.set_yticks(y,[r['label']+(' *' if r['group']==selection['selected'] else '') for r in summary]);ax.invert_yaxis()
    ax.set_xlim(0,1);ax.set_xlabel('AUROC (macro over R01–R04)');ax.set_title('45 | Frozen larger visual encoders, identical observations',loc='left',pad=18)
    ax.legend(loc='lower right',frameon=False);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
    fig.text(.025,.025,'* Selected using normal-only synthetic proxy. Mobile LoRA error bars: 3-seed SD, not CI.\n31,550 valid development frames; 1,912 mismatched-label frames excluded. Native resolutions differ.',fontsize=9,color='#50565d')
    fig.tight_layout(rect=[0,.09,1,1]);fig.savefig(OUT/'encoder_comparison.png',dpi=180);plt.close(fig)
    chosen=['B','C','clip_L','siglip2_L','dinov2_L'];colors=['#636b74','#326b9b','#c08027','#78803c','#b8668b'];marks=['o','s','^','D','X']
    plotted=[r for r in process if r['group'] in chosen]
    xmax=max(100*(r['metrics']['normal_fpr']['mean']+(r['metrics']['normal_fpr']['sample_std'] or 0)) for r in plotted)
    ymax=max(100*(r['metrics']['anomaly_recall']['mean']+(r['metrics']['anomaly_recall']['sample_std'] or 0)) for r in plotted)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,scene in zip(axes.flat,config()['scenes']):
        for g,color,marker in zip(chosen,colors,marks):
            r=next(r for r in process if r['scene']==scene and r['group']==g);m=r['metrics']
            ax.errorbar(100*m['normal_fpr']['mean'],100*m['anomaly_recall']['mean'],
                xerr=100*(m['normal_fpr']['sample_std'] or 0),yerr=100*(m['anomaly_recall']['sample_std'] or 0),
                fmt=marker,color=color,markersize=8,capsize=3,label=LABELS[g])
        ax.set_title(scene,loc='left');ax.set_xlim(0,max(1,xmax*1.12));ax.set_ylim(0,max(1,ymax*1.12));ax.grid(alpha=.16)
    fig.supxlabel('Normal frame false-positive rate (%)');fig.supylabel('Anomalous frame recall (%)')
    fig.suptitle('45 | Alarm trade-offs at each model’s own normal q99',x=.06,ha='left')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.025),fontsize=10)
    fig.text(.06,.012,'Historical Mobile LoRA: mean ± 3-seed SD. Separate thresholds; no matched-FPR superiority claim.',fontsize=9,color='#50565d')
    fig.tight_layout(rect=[.02,.14,1,.95]);fig.savefig(OUT/'alarm_tradeoffs.png',dpi=180);plt.close(fig)
    print('Report tables and 2 figures generated from validated results',flush=True)


if __name__=='__main__':main()
