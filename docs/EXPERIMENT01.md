# 실험 01 — 기본 파이프라인 구현

## 질문과 범위

정상 영상에서 로컬 VLM이 제안한 객체와 공정 상태를 이용해 검출·추적·정상 subspace·공정 점수를 연결할 수 있는가?

R01의 정상 학습 34개 영상(고정 fit/calibration 분할)과 테스트 15개 영상 전부를 사용한다. 결과는 이 장면과 seed 42에 한정한다. 선택 이유는 실제 영상이면서 라벨 길이 불일치가 없고 초기 전체 파이프라인 검증 비용이 작기 때문이다. 테스트 성능을 기준으로 장면을 선택하지 않았다.

## 고정 구현

1. 정상 FIT 영상 2개의 12프레임을 Qwen3.8-27B-Q8_0에 입력한다. 객체 역할 1–3개, 관측 가능한 phase 2–4개, 정상 순서 후보를 JSON으로 받는다. [입력 경로·prompt·최종 응답](../results/experiment01/process_discovery.json)을 보존한다. 테스트나 calibration 영상으로 prompt를 조정하지 않는다.
2. GroundingDINO tiny에 객체 noun phrase를 입력하고 4프레임마다 검출한다. 점수 임계값은 0.25, 역할별 NMS IoU는 0.5, 최대 3개 bbox를 유지한다. text phrase와 vocabulary의 단어 Jaccard overlap으로 역할을 대응한다.
3. 역할별 Hungarian IoU 매칭으로 tracking한다. IoU 0.2, 마지막 관측 이후 최대 2개 sampling interval을 허용한다. 이는 재식별 모델이 아니며 ID switch가 발생할 수 있다.
4. frozen CLIP ViT-B/32로 전체 프레임과 crop의 정규화된 512차원 특징을 추출한다. 생성형 VLM은 test-time에 호출하지 않는다.
5. 전체 프레임 CLIP 특징과 phase 설명의 text embedding cosine similarity를 사용한다. 현재/과거 최대 3샘플의 유사도를 평균한 뒤 argmax한다. phase GT 없이 얻은 proxy이다.
6. 정상 FIT 특징으로 `(객체 역할, phase)`별 PCA를 적합한다. explained variance 목표 0.95, 최대 rank 32이다. 그룹 표본이 10개 미만이면 같은 역할의 전체 phase PCA로 fallback하며, 역할 전체도 부족하면 full-frame PCA로 fallback한다. 최대 rank 제약으로 목표 분산에 미달할 수 있다.
7. 관측 phase 전이 빈도를 Laplace alpha=1로 평활화한다. `-log P(next|previous)`에 VLM 순서의 self/next/cycle 전이를 벗어나면 1을 더해 process raw score를 계산한다. 첫 샘플에는 이전 phase가 없어 raw score 0이다. 명시적인 duration 모델은 아직 없다.
8. NORMAL calibration 영상에서 역할별 PCA residual과 process score의 empirical percentile을 구한다. 동점은 midrank를 사용한다. visual score는 전체 프레임 및 검출 객체 percentile의 최댓값, 최종 score는 visual/process의 0.5/0.5 평균이다.
9. calibration의 최종 score q99를 경보 임계값으로 사용하며 `score > threshold`일 때 경보한다. 테스트 영상별 정규화나 테스트 기반 threshold 선택은 하지 않는다.
10. 4프레임 간격 점수를 이전 점수 유지 방식으로 원 프레임에 확장한다. 미래 샘플을 이용하지 않는다. 전체 장면의 프레임을 합쳐 AUROC와 average precision을 계산한다.

## 구현 범위의 한계

- bbox 위치·크기·track은 보존하지만 baseline의 visual score는 CLIP appearance residual이다. 이동량/관계 특징의 별도 scoring은 아직 없다.
- tracking은 객체 ID와 localization 후보 연결에 사용한다. PCA는 track ID가 아니라 역할별로 공유한다.
- 검출 없는 객체는 crop score가 없다. full-frame branch는 유지하지만 실제 누락과 detector 실패를 구분하지 않는다.
- bbox가 출력되어도 localization accuracy가 검증된 것은 아니다. 현재 데이터에 객체 GT가 없다.
- 논문의 SubspaceAD와 backbone·patch/crop 단위가 다르다. 원 논문 재현 성능이라고 부르지 않는다.
- latency는 특징 추출 단계의 별도 측정이다. 모델 loading, Qwen discovery, PCA fit, 최종 scoring, 실시간 입력 대기열을 포함한 end-to-end FPS가 아니다.
- 저장된 객체 score는 `object_detection_indices`를 통해 동일 파일의 bbox/track에 대응한다. `-1`은 전체 프레임 branch이다.

## 재현

```bash
export PYTHONPATH=src
.venv/bin/python scripts/download_models.py
# 별도 터미널에서: bash scripts/start_local_vlm.sh
.venv/bin/python scripts/discover_process.py --data-root /path/to/IPAD_dataset
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/extract_features.py \
  --data-root /path/to/IPAD_dataset
.venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset
.venv/bin/python scripts/plot_experiment.py --experiment 01
```

정확히 같은 vocabulary로 재현하려면 저장소의 `process_discovery.json`을 사용하고 discovery 재생성을 생략한다. temperature 0.2와 llama.cpp 기본 seed를 사용한 재생성은 동일 응답을 보장하지 않는다. feature cache는 config/process/model revision signature가 다르면 재사용을 거부한다.

## 결과와 Recommended improvements

R01 전체 실행을 완료했다. 결과는 [metrics.json](../results/experiment01/metrics.json), [시퀀스별 결과](../results/experiment01/per_sequence.csv)와 README에 기록했다. Combined AUROC 0.5371 / AP 0.3636이며, 중앙 phase의 FIT 점유율이 0인 실패에 따라 실험 02의 공간적 phase 추정을 설계했다.
