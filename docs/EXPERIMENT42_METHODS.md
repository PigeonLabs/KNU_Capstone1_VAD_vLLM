# 실험42 구현·재현 명세

## 모듈

- `src/ipad_vad/learned_association.py`: 512→8→512 residual metric head, 초기identity; IoU 우선의 causal role-gated tracker.
- `prepare_association42.py`: 정상pair후보와image/hash/box감사자료. 고정된split과41sourcehash를검증. TrackID는pair생성정답으로사용하지않음.
- `review_association42.py`, `finalize_pairs42.py`: 실제Codex시각검수결정을기록. 같은물체여부를자동판정하는스크립트가아님. 원본이미지·contactsheet는ignoredartifacts에유지.
- `train_association42.py`: train70/val19만 gradient/checkpoint에사용; cal/test미사용. 소스/config/감사결정과featurehash를optimizer첫step전에고정.
- `calibrate_association42.py`: 선택된checkpoint이후normalcalibrationnegative만으로role별appearancegate를고정.
- `extract_association42.py`: frozen41detector/featurecache에서tracker를재실행. Phase는원래auxiliaryCLIPfeature와기존phase모델로다시계산. 8개anomalyencoderfeature는재사용하고metadata만overlay. OriginalIoU/phasecache를전체정상·test에서정확히재현.
- `experiment42_normal.py`: IoU/raw/learned 각공정·8visualarm의통계모델을재적합. Control416model/score파일이41과정확히같음. Raw/learned의64full+352holdout도모두동일. Holdout은q99/CDF만제외한조건부진단임을기록.
- `evaluate_association42.py`: 64모델을다시구축하여고정normal모델과비교. 1,056testprediction을라벨전에고정한뒤AUROC/AP·경보·구간/paired41변화를계산.
- `diagnose_association42.py`, `validate_association42.py`, `report_association42.py`: 연결기회·추가연결시각검수자료,독립metric/이벤트/정렬검증,표·그림.

## 목적함수

`a,b`는정규화된동결CLIPcropfeature, `z(x)=normalize(x+B(Ax))`다. Positive항은`mean(1-z(a)·z(b))`, negative항은`mean(relu(z(a)·z(b)-0.5)^2)`다. Preservation은두endpoint에서`||z(x)-x||²`의평균이며가중치0.1이다. Trainbatch는positive/negative각128개여서preservation도1:1균형이다. Validationpositive는scene-rolemacro,negative는전체평균,preservation은positive전체평균/negative전체평균을1:1평균한다. Train기록도같은evaluationobjective로계산한다. Epoch0을포함한최저val선택; testmetric선택없음.

IoU매칭은기존threshold0.2의Hungarian을그대로먼저수행하며invalidedge를assignment전에막는다. 남은old/current노드만appearance후보가된다. Role·age·confidence·geometry·cosine·양방향ambiguity를통과한edge에Hungarian을적용한다. 유효IoU매칭을appearance가덮어쓰지않는다. 지원없는role/gate>1은정확히기존IoU동작이다. 각sample하나의track에두검출이할당되지않으며없는검출을출력하지않는다.

## 재현 순서

기존실험40/41의localfeaturecache와weight가필요하다. 공개repo에는이파일들을포함하지않는다. 같은completedoutput을덮어쓰지않게stageguard를두었다. 재실험은별도experimentdirectory/protocol을사용해야한다.

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/prepare_association42.py
# contact sheets를실제로검수한뒤저장된결정과대조
PYTHONPATH=src:scripts .venv/bin/python scripts/review_association42.py
PYTHONPATH=src:scripts .venv/bin/python scripts/finalize_pairs42.py
PYTHONPATH=src:scripts .venv/bin/python scripts/train_association42.py
PYTHONPATH=src:scripts .venv/bin/python scripts/calibrate_association42.py
PYTHONPATH=src:scripts .venv/bin/python scripts/extract_association42.py --split training
PYTHONPATH=src:scripts .venv/bin/python scripts/experiment42_normal.py
PYTHONPATH=src:scripts .venv/bin/python scripts/extract_association42.py --split testing
PYTHONPATH=src:scripts .venv/bin/python scripts/evaluate_association42.py
PYTHONPATH=src:scripts .venv/bin/python scripts/diagnose_association42.py
PYTHONPATH=src:scripts .venv/bin/python scripts/validate_association42.py
PYTHONPATH=src:scripts .venv/bin/python scripts/report_association42.py
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q
```

학습GPU는RTX PRO6000Blackwell,torch2.10,FP32/noTF32,seed42다. Head학습+validation3.70초/peakallocated0.271GiB. Frozenfeature에서normal/testreplay+sourcehashIO10.48/17.16초로측정했으며detector·encoderinference를포함하지않는다. End-to-end FPS나이전실험과의처리속도비교로사용하지않는다. 로컬VLM재호출은필요하지않았다.
