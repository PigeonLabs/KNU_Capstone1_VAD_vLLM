"""Trace shared-CDF interference in an unchanged entry-context raw model."""
import json
from pathlib import Path
import numpy as np
from ipad_vad.context_dwell import ContextDwell,observed_entry_context


def load(path):
    with np.load(path,allow_pickle=False) as f:return dict(f)


def main():
    ns=['12_support','12'];cfg={n:json.loads(Path(f'configs/experiment{n}.json').read_text()) for n in ns};scene=cfg['12']['scene']
    split=json.loads(Path('results/stage00/splits.json').read_text())[scene]
    data={group:[load(Path(f'artifacts/experiment12/features/{scene}/training_{seq}.npz')) for seq in split[group]] for group in ['fit','calibration']}
    models={}
    for n in ns:
        m=ContextDwell(**cfg[n]['normal_dwell'],**cfg[n]['dwell_context']);m.fit(data['fit']);m.calibrate(data['calibration']);models[n]=m
    np.testing.assert_array_equal(models['12_support'].durations[1],models['12'].context_durations[(0,1)])
    raw_values=[];scores={n:[] for n in ns}
    for d in data['calibration']:
        use=(observed_entry_context(d)==0)&(d['phases']==1)
        r=[models[n].raw(d)[0] for n in ns];np.testing.assert_array_equal(r[0][use],r[1][use]);raw_values.extend(r[0][use].tolist())
        for n in ns:scores[n].extend(models[n].score(d)[0][use].tolist())
    result={'scene':scene,'unchanged_context':'0->1','fit_durations_identical':True,'normal_calibration_raw_identical':True,'normal_calibration_observations':len(raw_values),
        'normal_raw_min_median_max':np.quantile(raw_values,[0,.5,1]).tolist(),
        'normal_percentile_min_median_max':{n:np.quantile(s,[0,.5,1]).tolist() for n,s in scores.items()},
        'mechanism':'Entry 0->1 raw model is unchanged, but the global dwell calibration reference includes raw scores of other contexts that changed. Therefore its percentile can change without changing its own duration model.',
        'limitation':'This is a mechanism check on the same normal calibration videos, not independent validation.'}
    Path('results/comparison11_12_support_12/calibration_interference.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
