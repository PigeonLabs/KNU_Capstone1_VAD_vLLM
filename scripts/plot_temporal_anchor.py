"""Plot normal-only continuity and observed phase support for experiment 18."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('results/experiment18')
    continuity = json.loads((root/'normal_temporal_continuity.json').read_text())['comparators']
    support = json.loads((root/'observation_conditioning.json').read_text())['normal_fit_phase_conditioning']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    keys = ['sign_changes', 'lost_runs_vs_ungated', 'one_sample_lost_runs_vs_ungated']
    x = np.arange(3)
    for offset, exp, color in [(-.18, '17', '#718096'), (.18, '18', '#137c8b')]:
        bars = axes[0].bar(x+offset, [continuity[exp][k] for k in keys], .36, label=f'Experiment {exp}', color=color)
        axes[0].bar_label(bars, padding=3)
    axes[0].set_xticks(x, ['Margin sign\nchanges', 'Lost relation\nruns', 'One-sample\nlost runs'])
    axes[0].set_ylim(0, 245)
    axes[0].set_ylabel('Count (normal FIT only)')
    axes[0].set_title('Fewer short gate interruptions')
    axes[0].legend(loc='upper right', frameon=False)
    observed = np.array([support[str(k)]['observed_samples'] for k in range(4)])
    missing = np.array([support[str(k)]['unobserved_samples'] for k in range(4)])
    a = axes[1].bar(np.arange(4), observed, label='Observed relation', color='#137c8b')
    b = axes[1].bar(np.arange(4), missing, bottom=observed, label='Held / initial phase', color='#dbaa5e')
    axes[1].bar_label(a, labels=['' if v < 10 else str(v) for v in observed], label_type='center', color='white')
    axes[1].annotate('3 observed', xy=(0, 3), xytext=(-.48, 195), fontsize=9, arrowprops={'arrowstyle': '->', 'color': '#137c8b'})
    axes[1].bar_label(b, label_type='center')
    axes[1].set_xticks(np.arange(4), [f'Latent {k}' for k in range(4)])
    axes[1].set_ylim(0, 1390)
    axes[1].set_ylabel('Normal FIT samples')
    axes[1].set_title('837 / 1,960 phases lack a current relation')
    axes[1].legend(loc='upper right', frameon=False)
    fig.suptitle('R04 development: temporal stability is not semantic accuracy', fontsize=13)
    fig.savefig(root/'temporal_anchor.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
