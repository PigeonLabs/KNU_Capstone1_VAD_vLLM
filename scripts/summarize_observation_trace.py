"""Summarize frozen normal-only diagnostics; no scoring or parameter selection."""
import csv,json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/experiment33')
def read(name):return json.loads((OUT/name).read_text())
def save(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def main():
    audit=read('candidate_audit.json');boundaries=read('gap_and_track_audit.json');episodes=read('episode_boundary_trace.json')['episodes'];summary={}
    for part,stats in audit['partitions'].items():
        sequences=[r['sequence'] for r in audit['sequences'] if r['partition']==part]
        gaps=[r for r in boundaries['gaps'] if r['partition']==part];changes=[r for r in boundaries['consecutive_observed_pair_changes'] if r['partition']==part];eps=[r for r in episodes if r['partition']==part];c=Counter()
        for seq in sequences:
            rows=json.loads((Path('artifacts/experiment33/traces')/f'{seq}.json').read_text())['frames']
            for row in rows:
                a=row['roles']['anchor'];t=row['roles']['target'];raw=a['candidates'];joint=[x for x in raw if x['joint_pass']];selected=[x for x in raw if x['selected']]
                c['anchor_missing_only']+=a['selected_detection_index']<0<=t['selected_detection_index'];c['target_missing_only']+=t['selected_detection_index']<0<=a['selected_detection_index'];c['both_missing']+=a['selected_detection_index']<0 and t['selected_detection_index']<0
                area_rejected=[x for x in raw if x['positive_area'] and x['semantic_pass'] and not x['area_pass']]
                c['has_semantic_positive_area_rejected_anchor']+=bool(area_rejected)
                c['no_anchor_but_positive_margin_area_rejected_available']+=not selected and bool(area_rejected)
                c['relation_missing_with_target_available_and_positive_margin_area_rejected_anchor']+=not selected and bool(area_rejected) and t['selected_detection_index']>=0
                c['anchor_multiple_joint_candidates']+=len(joint)>1
                if selected:
                    c['selected_anchor_samples']+=1
                    better=[x for x in joint if x['gate_margin']>selected[0]['gate_margin']]
                    c['selected_anchor_has_higher_margin_eligible_alternative']+=bool(better)
                    c['retained_anchor_has_higher_margin_eligible_alternative']+=bool(better) and a['selection_event']=='retained'
                    c['selected_anchor_nonpositive_raw_positive_temporal']+=selected[0]['raw_margin']<=0<selected[0]['gate_margin']
        assert sum(c[k] for k in ['anchor_missing_only','target_missing_only','both_missing'])==stats['samples']-stats['valid_samples']
        summary[part]={'samples':stats['samples'],'valid_samples':stats['valid_samples'],'valid_rate':stats['valid_samples']/stats['samples'],'diagnostics':dict(c),'missing_gaps':len(gaps),'complete_gaps':sum(r['complete_gap'] for r in gaps),'same_pair_reacquired_complete_gaps':sum(r['same_pair_reacquired'] is True for r in gaps),'changed_pair_reacquired_complete_gaps':sum(r['same_pair_reacquired'] is False for r in gaps),'consecutive_pair_changes':dict(Counter(','.join(r['changed_roles']) for r in changes)),'change_selection_events':dict(Counter(role+'|'+r['current']['roles'][role]['selection_event'] for r in changes for role in r['changed_roles'])),'known_entry_censor_missing_boundary_reasons':dict(Counter(r['at_end_boundary']['failure_combination'] for r in eps if r['status']=='right_censored' and r['end_reason']=='relation_missing')),'unknown_reacquisition_previous_reasons':dict(Counter(r['before_start']['failure_combination'] for r in eps if r['start_reason']=='reacquired'))}
    save('diagnostic_summary.json',{'normal_only':True,'posthoc_metadata_summary':True,'candidate_margin_is_not_role_ground_truth':True,'potential_area_recovery_is_not_an_executed_intervention':True,'partitions':summary})
    with (OUT/'candidate_reasons.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['partition','role','reason','samples','denominator','fraction'])
        for part,s in audit['partitions'].items():
            for role,counts in s['role_reasons'].items():
                for reason,n in counts.items():w.writerow([part,role,reason,n,s['samples'],n/s['samples']])
    reasons=['no_raw_candidate','no_positive_area','semantic_excluded','area_excluded','no_joint_candidate','candidate_available'];labels=['No cached candidate','No positive area','Semantic exclusion','Area exclusion','No joint candidate','Candidate available'];colors=['#cfdae8','#bcc8d6','#849bb8','#4a709e','#244c7b','#16395d']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,(part,stats) in zip(axes,audit['partitions'].items()):
        for j,role in enumerate(['anchor','target']):
            left=0
            for reason,color in zip(reasons,colors):
                n=stats['role_reasons'][role][reason];pct=100*n/stats['samples'];ax.barh(j,pct,left=left,color=color,edgecolor='white',height=.5)
                if pct>=6:ax.text(left+pct/2,j,f'{n}',ha='center',va='center',color='white' if reason not in reasons[:2] else '#18314f',fontsize=10)
                left+=pct
        ax.set(xlim=(0,100),yticks=[0,1],yticklabels=['Anchor','Target'],xlabel='Share of sampled frames (%)',title=f'{part.upper()} | n={stats["samples"]:,} samples');ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,labels)],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.065))
    fig.suptitle('Experiment 33: normal candidate availability by role',fontsize=15)
    fig.text(.5,.025,'Disjoint diagnostic categories; counts label segments. Availability is not semantic accuracy.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.08,right=.98,top=.82,bottom=.3,wspace=.16);fig.savefig(OUT/'candidate_availability.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
