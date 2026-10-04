# 실험 04 — 정상 phase별 진행량을 공정 점수에 추가

## 직전 결과와 변경 이유

실험 03의 제품 관측 개선 후에도 Visual AUROC 0.6514보다 Combined 0.5814가 낮았다. 전이 빈도는 같은 phase에서 정상적으로 진행하는 제품과 멈춘 제품을 명시적으로 구별하지 않는다. 실험 03의 1순위 권고에 따라 실제 진행량 신호를 추가했다. [실행 전 계획과 고정 명세](EXPERIMENT04_PLAN.md)

## 방법과 고정 조건

1. 실험 03의 bbox·track·encoder 특징·phase map·phase 배열을 그대로 사용한다. 선택된 제품 anchor가 인접 sampled 시점에 모두 존재하고 track ID가 같을 때만 진행량을 계산한다.
2. `v_t = (x_t - x_(t-1)) / (frame_t - frame_(t-1))`. x는 정규화된 bbox 중심이다. 단위는 정규화 좌표/원본 프레임이며 FPS를 가정하지 않는다.
3. 정상 FIT의 phase별 median과 `max(1.4826*MAD, 1e-6)`를 적합하고 절대 표준화 편차를 계산한다. 표본 10개 미만 phase는 pooled FIT 모델로 fallback한다. 이번 데이터의 모든 phase에는 충분한 표본이 있었다.
4. 정상 calibration의 **유효한 진행량 residual만** 모아 기존과 같은 midrank percentile로 변환한다. 진행량이 정의되지 않은 위치의 저장용 0은 학습·보정·점수에 사용하지 않는다.
5. 유효 진행량이 있으면 `process = max(기존 전이 percentile, 진행량 percentile)`, 없으면 기존 전이 점수를 유지한다. 최종 `0.5*visual + 0.5*process`와 정상 calibration q99 정책은 유지한다. 새 결합 결과에 대해 q99를 다시 계산한다.

정상 FIT 진단에서 1,556개 sampled 시점 중 1,432개(92.03%)가 유효했고 phase별 표본은 423/486/523개였다. 위 모델과 max 결합을 고정한 뒤 테스트 지표를 계산했다. detector·VLM을 재실행하거나 테스트 점수로 가중치·threshold를 탐색하지 않았다. max는 사전 고정한 OR 결합 규칙이며 최적 결합이라는 주장은 아니다.

## 이번 실험 결과

R01 정상 fit 27개/calibration 7개, 테스트 15개 영상, seed 42다. 테스트 3,685프레임 중 이상은 1,254프레임이다. 실험 03과 평가 라벨 배열 및 원본 특징 배열의 동일성을 검사했다.

| 지표 | 실험 03 | 실험 04 |
|---|---:|---:|
| Visual AUROC | 0.6514 | 0.6514 |
| Visual AP | 0.4260 | 0.4260 |
| Process AUROC | 0.5388 | 0.6738 |
| Process AP | 0.3603 | 0.6167 |
| Combined AUROC | 0.5814 | 0.6846 |
| Combined AP | 0.4018 | 0.6298 |
| 정상 q99에서 테스트 정상 오탐률 | 2.96% | 9.95% |
| 정상 q99에서 테스트 이상 recall | 3.83% | 41.95% |

실험 04 정상 q99 임계값은 0.9767이다. 두 실험은 각자의 정상 calibration에서 구한 임계값을 적용했으므로 동일 테스트 오탐률 비교가 아니다. 정상 calibration sampled alarm rate는 0.98%였지만 테스트 정상 프레임 오탐률은 9.95%다. 정상 calibration의 q99가 새 정상 데이터에서 1% 오탐을 보장하지는 않는다.

| 진행량 가용성 | 유효/전체 sampled 시점 | 비율 |
|---|---:|---:|
| 정상 FIT | 1,432 / 1,556 | 92.03% |
| 정상 calibration | 378 / 410 | 92.20% |
| 테스트 | 870 / 927 | 93.85% |

점수를 이전 관측값으로 확장한 dense 기준 유효 구간은 3,478/3,685프레임(94.38%)이다. 이 **유효 구간만** 대상으로 진행량 단독 AUROC는 0.6451, AP는 0.6212였다. 여기에는 이상 1,245프레임이 포함되므로 전체 프레임 지표와 같은 평가 대상이 아니다. 나머지 207프레임은 실험 03의 Combined 점수와 정확히 같았다.

![실험 03–04 비교](../results/comparison03_04/comparison.png)

[전체 지표](../results/experiment04/metrics.json) · [영상별 결과](../results/experiment04/per_sequence.csv) · [정상 FIT 사전 진단](../results/experiment04/normal_progress_diagnostic.json) · [관측 및 source hash](../results/experiment04/progress_features.json) · [오류 진단](../results/experiment04/error_diagnosis.json)

## 실험 결과의 의의

동일한 외형·검출·phase에서 진행량만 추가해 Combined AUROC +0.1032, AP +0.2279와 이상 recall 상승을 관측했다. Visual 점수 및 원래 전이 점수는 모든 테스트 영상에서 동일했다. 따라서 이번 비교는 외형 외에 시간적 진행량이 추가 신호를 제공할 수 있다는 구현상의 근거다. phase 조건·median/MAD·max 결합 각각의 독립적 기여를 분리한 증거는 아니다.

논문에서는 정상 영상에서 얻은 객체/공정 후보, 정상 경로로 제한한 검출, phase별 외형과 진행량을 연결하고 관측이 불가능하면 진행량을 사용하지 않는 파이프라인을 구체적으로 설명할 수 있다. 구성 요소의 연결과 실패 분석이 현재 확보한 가치이며, 기존 방법과 구별되는 알고리즘 novelty나 일반화가 입증됐다고 주장하지 않는다.

## 보완할 점

- **오탐 증가:** 정상 오탐률 2.96%→9.95%. 이상 recall 상승만으로 전체 시스템이 개선됐다고 결론 내리지 않는다.
- **짧은 경보 변동:** 정상 오탐 구간 41개 중 30개(73.17%)가 4프레임 이하다. 오탐 구간 median 4프레임, 최대 32프레임이다. bbox jitter 또는 단일 간격 진행량의 변동이 원인일 수 있지만 아직 입증하지 않았다.
- **짧은 이상 손실 위험:** 실제 이상과 겹치는 경보 구간 15개 중 7개(46.67%)도 4프레임 이하다. 경보를 단순히 짧다고 제거하면 유효 탐지가 사라질 수 있다. 이 구간 통계는 GT 이상 이벤트별 recall이 아니다.
- **정상 분포와 결합:** phase 0·2 진행량 분포는 phase 1보다 넓다. 단일 median/MAD가 여러 정상 속도 조건을 모두 표현하는지, max가 정상 극단값을 과도하게 증폭하는지 미확인이다.
- **미관측과 범위:** 진행량이 없으면 기존 전이 점수로 돌아간다. 실제 제품 누락을 새 모듈이 해결하지 않는다. 독립 bbox/phase GT도 없으며, 반복 관찰한 R01은 개발용이다. 다른 장면·seed·고정 오탐률·통계적 유의성 검증은 하지 않았다.

## 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **여러 연속 관측으로 진행량 추정:** 동일 track의 더 긴 인과적 간격에서 변위를 계산 | 오탐 구간의 73.17%가 한 sampling 간격 이하 | detector·특징·phase·점수 결합 고정. 사전 고정한 한 가지 시간 간격만 비교하고 유효 관측률·오탐/recall·원본 프레임 기준 탐지 지연을 보고. 짧은 이상 손실 위험 확인 |
| 2 | **phase 조건의 필요성 분리 검증:** 동일 진행량의 phase별 모델과 pooled 모델 비교 | 진행량 추가는 효과가 있었지만 phase 조건의 독립 기여는 미분리 | 다른 모듈을 고정하고 조건화만 바꿔 복잡성의 필요성을 검증. 성능이 비슷하면 단순한 구성을 채택할 근거로 사용. novelty 주장을 위해 유리한 결과만 선택하지 않음 |
| 3 | **미사용 장면 적용성과 검증 분리:** 다른 정상 공정 영상으로 discovery·경로·진행량 적합 | R01 반복 개발 결과에 한정, 왼쪽→오른쪽 공간 모델의 범용성 미검증 | 라벨 정합성이 확보된 장면을 테스트 성능과 무관하게 선택하고 정상 데이터로만 설정 적합. 적용 실패도 보고하며 장면별 데이터/grammar 차이를 구분 |

현재 1순위를 [실험 05 계획](EXPERIMENT05_PLAN.md)으로 선택한다. 2·3순위는 확정된 이후 실험 번호가 아니며 다음 결과에 따라 재평가한다.

## 재현과 검증

실험 03 특징 및 결과가 먼저 필요하다. 이번 실험은 저장된 로컬 특징으로 수행하며 추가 VLM/GPU 추론이 없다. 처리 속도는 별도로 측정하지 않았다.

```bash
export PYTHONPATH=src
.venv/bin/python scripts/prepare_progress.py --normal-diagnostic-only
.venv/bin/python scripts/prepare_progress.py
.venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset \
  --config configs/experiment04.json
.venv/bin/python scripts/diagnose_motion_errors.py --experiment 04 --baseline 03
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/plot_experiment.py --experiment 04
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/compare_runs.py --experiments 03 04
.venv/bin/python -m pytest -q
```

17개 테스트 통과. 미래 위치 변경이 과거 진행량에 영향을 주지 않는지, frame 간격으로 나누는지, ID 변경/누락을 0속도로 취급하지 않는지, invalid sentinel을 적합·보정에서 제외하는지, max 결합 및 q99 재보정이 정확한지 검사했다. 49개 특징 캐시와 15개 테스트 예측에서 변경 범위·라벨·기존 branch 보존을 확인했다. [검증 기록](../results/experiment04/validation.json)
