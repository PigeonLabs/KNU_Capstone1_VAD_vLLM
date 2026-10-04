import json
import pytest
from ipad_vad.experiment import discovery_path,load_process


def test_new_scene_cannot_silently_reuse_r01_discovery(tmp_path):
    with pytest.raises(ValueError):discovery_path({'scene':'R03'})
    p=tmp_path/'process.json'
    p.write_text(json.dumps({'scene':'R01','sources':['R01/training/frames/01/0.jpg']}))
    with pytest.raises(ValueError):load_process({'scene':'R03','process_discovery':str(p)})


def test_discovery_requires_normal_scene_sources_and_complete_order(tmp_path):
    p=tmp_path/'process.json';cfg={'scene':'R03','process_discovery':str(p)}
    d={'scene':'R03','sources':['R03/training/frames/01/0.jpg'],'process':{'objects':[{'id':'truck'}],'phases':[{'id':'a'},{'id':'b'}],'normal_order':['a','b']}}
    p.write_text(json.dumps(d));assert load_process(cfg)==d['process']
    d['process']['normal_order']=['a','a','b'];p.write_text(json.dumps(d))
    with pytest.raises(ValueError):load_process(cfg)
    d['sources']=['R03/testing/frames/01/0.jpg'];p.write_text(json.dumps(d))
    with pytest.raises(ValueError):load_process(cfg)
