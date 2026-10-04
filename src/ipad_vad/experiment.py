"""Scene-specific process hypotheses; prevent implicit cross-scene reuse."""
import json
from pathlib import Path


def discovery_path(config):
    if 'process_discovery' in config:return Path(config['process_discovery'])
    if config['scene']=='R01':return Path('results/experiment01/process_discovery.json')
    raise ValueError('A new scene requires an explicit process_discovery path')


def load_process(config):
    record=json.loads(discovery_path(config).read_text())
    if record['scene']!=config['scene']:raise ValueError('Discovery scene does not match experiment scene')
    if not record['sources'] or any(not s.startswith(config['scene']+'/training/frames/') for s in record['sources']):
        raise ValueError('Discovery must reference this scene normal training frames')
    process=record['process'];roles=[o['id'] for o in process['objects']];phases=[p['id'] for p in process['phases']]
    if len(roles)!=len(set(roles)) or len(phases)!=len(set(phases)):raise ValueError('Duplicate process IDs')
    order=process['normal_order']
    if len(order)!=len(phases) or set(order)!=set(phases):raise ValueError('Normal order must cover each phase once')
    return process
