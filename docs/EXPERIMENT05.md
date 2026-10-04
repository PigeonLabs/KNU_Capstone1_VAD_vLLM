# 실험 05 — 여러 연속 관측에 걸친 진행량

## 직전 결과와 선택한 변경

실험 04에서는 진행량 신호로 이상 탐지가 늘었지만 정상 오탐률도 9.95%로 상승했다. 오탐 경보 구간 41개 중 30개가 4프레임 이하였으므로 단일 관측 간격의 변동에 민감할 가능성을 검증했다. [실행 전 계획](EXPERIMENT05_PLAN.md)의 1개 후보만 실행했다.

진행량을 `v_t = (x_t - x_(t-3)) / (frame_t - frame_(t-3))`로 교체했다. sampling 4프레임에서 12프레임 변위다. 현재와 과거의 연속 4개 anchor가 모두 존재하고 같은 track ID일 때만 유효하다. 중간 누락 또는 ID 변경이 있으면 사용할 수 없으며 기존 전이 점수로 돌아간다. 미래 위치나 경보 구간 길이를 이용해 사후 수정하지 않는다.

detector, ROI, CLIP 특징, phase map/배열, 정상 분할, phase별 median/MAD, max 결합, 0.5/0.5 최종 가중치는 유지했다. 변경된 진행량으로 정상 모델과 calibration q99를 다시 적합했다. 다른 시간 간격의 성능을 탐색하지 않았다.

## 이번 실험 결과

R01·seed 42, 정상 fit 27개/calibration 7개, 테스트 15개 영상의 3,685프레임(이상 1,254개)을 평가했다. 각 모델의 정상 calibration q99 기준이며 동일 테스트 오탐률 비교가 아니다.

| 지표 | 실험 04 | 실험 05 |
|---|---:|---:|
| Visual AUROC | 0.6514 | 0.6514 |
| Visual AP | 0.4260 | 0.4260 |
| Process AUROC | 0.6738 | 0.6889 |
| Process AP | 0.6167 | 0.6502 |
| Combined AUROC | 0.6846 | 0.6992 |
| Combined AP | 0.6298 | 0.6584 |
| 정상 q99 기준 테스트 정상 오탐률 | 9.95% | 5.92% |
| 정상 q99 기준 테스트 이상 recall | 41.95% | 42.26% |

실험 05 임계값은 0.9846, 정상 calibration sampled alarm rate는 0.98%다. Visual 및 기존 전이 점수는 실험 04와 정확히 같다.

![실험 04–05 비교](../results/comparison04_05/comparison.png)

| 관측 및 경보 진단 | 실험 04 | 실험 05 |
|---|---:|---:|
| 정상 FIT 유효 진행량 / 전체 samples | 1,432 / 1,556 | 1,374 / 1,556 |
| 정상 calibration 유효 진행량 / 전체 samples | 378 / 410 | 362 / 410 |
| 테스트 유효 진행량 / 전체 samples | 870 / 927 (93.85%) | 838 / 927 (90.40%) |
| 테스트 유효 진행량 / dense 프레임 | 3,478 / 3,685 (94.38%) | 3,350 / 3,685 (90.91%) |
| 정상 오탐 프레임 수 | 242 | 144 |
| 정상 오탐 경보 구간 수 | 41 | 27 |
| 정상 오탐 구간 median / 최대 길이 | 4 / 32프레임 | 4 / 16프레임 |

실험 05 진행량 단독의 **유효 dense 구간만** 평가한 AUROC/AP는 0.6669/0.6630이다. 이 구간은 3,350프레임(이상 1,229개)으로 전체 평가와 대상이 다르며, 실험 04의 유효 구간과도 달라 단독 지표를 동일 대상 비교로 해석하지 않는다. 나머지 335프레임은 진행량 모듈 적용 전인 실험 03의 Combined 점수와 같다.

### 이상 구간 탐지와 지연

GT의 연속 양성 구간 8개에 대해, 고정된 정상 q99를 넘는 첫 경보까지의 원본 프레임 수를 계산했다. 탐지란 구간 안에 경보가 하나라도 있는 경우다. 프레임 점수를 point adjustment하지 않았다. 미탐은 delay null로 보존하고 분모에서 제외하지 않는다.

| 영상 | 실험 04 첫 경보 지연 | 실험 05 첫 경보 지연 |
|---|---:|---:|
| R01_01 | 13 | 13 |
| R01_02 | 50 | 50 |
| R01_03 | 31 | 31 |
| R01_04 | 18 | 18 |
| R01_05 | 0 | 2 |
| R01_06 | 66 | 66 |
| R01_07 | 71 | 71 |
| R01_08 | 46 | 46 |

- 두 실험 모두 8/8구간에서 경보가 있었고 미탐 구간은 0개다. 이것이 이상 프레임 전체를 탐지했다는 뜻은 아니며 프레임 recall은 약 42%다.
- 첫 경보 지연 중앙값은 두 실험 모두 38.5프레임, 최대 71프레임이다. 대응 구간별 지연 변화 중앙값은 0이며 R01_05만 2프레임 증가했다.
- 실험 04의 R01_05는 이상 시작 전부터 경보가 켜져 있었다. 이 경우의 delay 0을 정확한 onset 탐지로 해석하지 않는다. 실험 05에는 이런 사례가 없었다.
- 12프레임 이하 GT 이상 구간은 **0개**다. 따라서 짧은 이상을 보존했는지는 이번 데이터로 검증할 수 없다. FPS가 없으므로 초 단위 지연을 주장하지 않는다.

[전체 지표](../results/experiment05/metrics.json) · [영상별 결과](../results/experiment05/per_sequence.csv) · [관측 진단](../results/experiment05/progress_features.json) · [오류 진단](../results/experiment05/error_diagnosis.json) · [이벤트별 지연 및 미탐 기록](../results/comparison04_05/events.json)

## 실험 결과의 의의

동일한 제품 관측과 phase에서 시간 간격을 늘린 결과, 정상 오탐 프레임 242→144와 Combined AUROC +0.0147, AP +0.0287을 관측했다. 이번 R01 비교에서는 프레임 recall과 8개 구간의 탐지 여부를 유지하면서 경보 변동을 줄이는 방향이었다. 이 결과만으로 bbox jitter가 원인이었다거나 12프레임 간격이 일반적으로 최적이라고 결론 내리지 않는다.

진행량을 정의할 수 있는 조건과 미관측 fallback을 명시하고, 단순 성능 외에 관측 가용성과 탐지 지연을 함께 비교하는 파이프라인 근거를 추가했다. 새로운 시계열 이론이나 알고리즘 novelty를 입증한 것은 아니며, phase 조건화가 꼭 필요한지도 아직 분리 검증하지 않았다.

## 보완할 점

- **관측 가용성 감소:** 연속 관측 요구로 테스트 sampled 유효 비율이 93.85%→90.40%로 줄었다. 시작 구간과 ID 단절 직후에는 새 시간 신호를 사용할 수 없다.
- **초기 탐지 지연:** 모든 GT 구간에 경보가 있어도 첫 경보 중앙값이 38.5프레임이다. 초기 이상을 즉시 포착하는 시스템으로 해석하지 않는다.
- **짧은 이상 검증 불가:** 이번 GT에는 12프레임 이하 이상이 없다. 짧은 경보 구간과 짧은 GT 이상 구간을 혼동하지 않는다.
- **잔여 오탐 및 범위:** 정상 q99임에도 테스트 정상 오탐률은 5.92%다. 단일 장면·seed의 반복 개발 결과로 최종 일반화 성능·통계적 유의성을 주장하지 않는다.
- **기여 분리 부족:** phase 조건·외형 subspace·전이·진행량·결합의 여러 구성 요소가 있다. 증가한 복잡성이 각각 필요한지 추가 ablation이 필요하다. 객체 부재 및 경로 밖 이상도 아직 해결하지 않았다.

## 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **진행량의 phase 조건 필요성 분리:** phase별 median/MAD 대신 pooled 정상 진행량 모델 비교 | 시간 신호·간격의 효과는 관측했지만 phase 조건의 독립 기여는 모름 | 진행량·유효 마스크·외형 subspace·전이·결합 규칙을 모두 고정하고 진행량 조건화만 제거. 성능·오탐·지연·모델 수를 비교해 필요한 복잡성을 판단 |
| 2 | **미사용 장면과 짧은 이상 검증:** 다른 공정/이상 길이에서 적용성 평가 | R01의 8개 이상 구간에는 짧은 이상이 없고 반복 개발에 사용됨 | 테스트 성능과 무관하게 장면을 선택하고 정상 영상으로만 적합. 자연 데이터와 인위적 스트레스 검사를 구분하며 라벨/시간 정합성 재확인 |
| 3 | **초기·미관측 구간의 이상 신호 보완:** 관측 누락·제품 부재·진행량 warm-up 실패 원인 진단 | 첫 경보 중앙값 38.5프레임, dense 335프레임에서 진행량 부재 | 구간별 지연과 누락의 관계를 먼저 확인한 뒤 초기 신호 변경. 정상 빈 벨트와 실제 제품 누락을 구분하고 미래 정보 사용 금지 |

다음 실험은 1순위의 조건화 제거 비교다. [실험 06 계획](EXPERIMENT06_PLAN.md)에 한 단계만 구체화했다. pooled가 비슷하거나 더 좋으면 단순화를 고려하고, 불리하면 조건화의 유용성과 비용을 기록한다. 성능을 높이기 위해 결과에 맞춰 조건을 고르지 않는다.

## 재현 및 검증

실험 03 특징이 필요하다. 이번 실험에는 추가 VLM/GPU 추론이 없고 실행시간은 따로 측정하지 않았다.

```bash
export PYTHONPATH=src
.venv/bin/python scripts/prepare_progress.py --config configs/experiment05.json --normal-diagnostic-only
.venv/bin/python scripts/prepare_progress.py --config configs/experiment05.json
.venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset \
  --config configs/experiment05.json
.venv/bin/python scripts/diagnose_motion_errors.py --experiment 05 --baseline 04
.venv/bin/python scripts/compare_event_delays.py --experiments 04 05
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/plot_experiment.py --experiment 05
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/compare_runs.py --experiments 04 05
.venv/bin/python -m pytest -q
```

20개 테스트 통과. 49개 캐시에서 기존 특징 보존, lag=1 재계산의 실험 04와 완전 일치, 긴 간격 유효 마스크가 짧은 간격의 부분집합임을 확인했다. 미래 위치 변경에 대한 인과성, 중간 누락/ID 변경 거부, 이벤트 미탐 분모·사전 경보·경계 censoring을 검사했다. 15개 예측에서 Visual·원래 전이·라벨 보존과 미관측 fallback을 검증했다. [검증 결과](../results/experiment05/validation.json)
