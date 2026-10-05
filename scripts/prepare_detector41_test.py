"""Build the test inventory only after all normal models are frozen; no labels."""
import json
from pathlib import Path
from ipad_vad.learned_detector import sha,write
from experiment41_normal import OUT,ART,verify_normal
from experiment41_phases import load_npz


def main():
    verify_normal();record=json.loads((OUT/'detector_testing_extraction.json').read_text());sequences=[];hashes={}
    for r in record['sequences']:
        path=ART/'detections/features'/r['scene']/f'testing_{r["sequence"]}.npz';assert sha(path)==r['sha256'];d=load_npz(path)
        hashes[str(path)]=sha(path);sequences.append({'scene':r['scene'],'sequence':r['sequence'],'source_frames':int(d['frame_count']),'views':{'global':len(d['indices']),'crop':len(d['boxes'])}})
    write(OUT/'test_data_manifest.json',{'normal_checkpoint_sha256':sha(OUT/'normal_models_checkpoint.json'),'test_source_feature_sha256':hashes,'sequences':sequences,'labels_opened':False})
    print('Test inventory prepared',len(sequences))


if __name__=='__main__':main()
