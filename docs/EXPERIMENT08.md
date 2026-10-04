# 실험 08 — 객체 관계 기반 잠재 상태

## 이번 실험 결과

**객체 관계로 phase를 바꾸자 Visual AUROC는 0.6968→0.7330으로 상승했지만, Combined AUROC는 0.6747→0.6361로 하락했다.** 정상 상태별 공정 점수의 보정 차이를 후속 검증 대상으로 확인했다. 이 실험은 R03 개발 장면에서 완료했으며, semantic phase 정답을 복원한 실험은 아니다.

### 질문과 변경 범위

실험 07에서 `carrying` phase는 모든 분할에서 0개였다. 전체 프레임–텍스트 유사도 대신 지게차–팔레트의 관계로 구별 가능한 정상 상태를 만들 수 있는지 확인했다. 비교 기준은 실험 07 기본 설정이며, 실험 07_motion의 진행량 모듈은 포함하지 않았다.

- 정상 FIT 18개 / calibration 4개 / 테스트 17개 영상, seed 42. 평가 12,005프레임 중 이상 5,068프레임이며 R03은 라벨 길이가 모두 일치한다.
- 실험 07의 detector·tracking·bbox·CLIP crop/full-frame 특징·4프레임 sampling·분할을 재사용했다. 기존 배열 중 phase만 변경하고, 그 phase에 의존하는 외형 PCA·전이·정상 calibration을 다시 적합했다.
- PCA variance 0.95, 최대 rank 32, 최소 phase 표본 10, 전이 Laplace 1, self/next/cycle 외 전이 penalty 1, Visual/Process 0.5/0.5 결합을 유지했다. 점수는 이전 sampled 값을 유지해 원본 프레임으로 확장했다.
- 각 모델의 정상 calibration q99를 사용했다. 테스트 라벨로 threshold나 가중치를 선택하지 않았다. 정상 진단 후 설정을 고정한 [평가 전 기록](../results/experiment08/pre_evaluation_protocol.json)을 남겼다.

### 관계 상태 구현

역할별 정상 FIT bbox 로그 면적의 `Q3 + 1.5 × IQR`로 상한을 정했다. 정규화 면적 상한은 지게차 0.687666, 팔레트 0.106046이며 FIT 후보의 각각 4.06%, 6.13%가 제외됐다. 이 제한은 **phase anchor 선택에만** 적용한다. 외형 branch는 기존 모든 bbox/crop을 유지한다. 후보 중 직전 track을 우선하고 confidence로 선택한다.

관계 descriptor는 팔레트 상대 중심 x/y(지게차 bbox 크기로 나눔), 로그 면적비, bbox IoU, 지게차 절대 중심 x/y의 6차원이다. 현재와 과거 최대 3개 유효 관측을 평균한다. 누락 시 history를 비우고 이전 상태를 유지하며, 첫 관측 전에는 latent_0을 사용한다. 누락은 별도 마스크로 남기고 군집 적합에서는 제외한다. track ID 변경 자체로 history를 비우지는 않는다.

정상 FIT median/IQR(최소 scale 0.05)로 정규화한 뒤 KMeans 4개(seed 42, n_init 10)를 적합했다. 상태 번호만 정상 FIT의 median 상대 프레임 위치 순으로 정렬했다. **테스트 시간·영상 길이·미래 관측은 상태 추정 입력에 사용하지 않는다.** latent_0..3에 idle/carrying 같은 행동 의미를 부여하지 않는다. 기존 4상태 self/next/cycle 구조의 수치 규칙은 유지했지만, 이 순서 prior가 관계 군집에 적합하다는 보장은 없다.

| sampled 관측 진단 | 정상 FIT | 정상 calibration | 테스트 |
|---|---:|---:|---:|
| 유효 관계 / 전체 | 3,136 / 3,142 | 703 / 703 | 3,007 / 3,007 |
| 유효 관계 비율 | 99.81% | 100% | 100% |
| latent_0 (누락 hold 포함) | 1,189 | 268 | 1,253 |
| latent_1 | 975 | 211 | 801 |
| latent_2 | 747 | 208 | 788 |
| latent_3 | 231 | 16 | 165 |

FIT 유효 군집 표본은 1,186 / 972 / 747 / 231개다. 높은 관측률과 네 군집의 support는 검출 recall 또는 의미 상태 정확도가 아니다.

### 탐지 결과

| R03 지표 | 실험 07: 텍스트 phase | 실험 08: 관계 latent phase |
|---|---:|---:|
| Visual AUROC | 0.6968 | 0.7330 |
| Visual AP | 0.6374 | 0.6778 |
| Process AUROC | 0.4980 | 0.5520 |
| Process AP | 0.4223 | 0.4693 |
| Combined AUROC | 0.6747 | 0.6361 |
| Combined AP | 0.5664 | 0.5496 |
| 정상 q99 threshold | 0.970741 | 0.966081 |
| 테스트 정상 오탐률 | 3.85% | 3.36% |
| 테스트 이상 프레임 recall | 5.56% | 10.36% |
| 경보가 발생한 GT 이상 구간 | 10 / 17 | 11 / 17 |
| 미탐 GT 이상 구간 | 7 | 6 |

![실험 07–08 비교](../results/comparison07_08/comparison.png)

GT 연속 이상 구간에 경보가 하나라도 발생하면 구간 탐지로 센다. point adjustment는 하지 않는다. 두 실험 모두 탐지한 9개 구간의 지연 변화 중앙값은 -104원본 프레임이다. 실험 08은 `R03_05 [348,407)`, `R03_09 [620,712)`를 새로 탐지하고 `R03_15 [620,692)`를 놓쳤다. 두 모델 모두 놓친 구간은 5개다.

탐지 구간만의 지연 중앙값은 241→150프레임이지만 대상 집합이 다르므로 이를 전체 지연 개선량으로 해석하지 않는다. 실험 08의 1개 구간은 이상 시작 전에 경보가 이미 켜져 있었고, 4개 GT 구간은 영상 왼쪽 경계에서 시작한다. 12프레임 이하 GT 구간은 0개다. 서로 다른 정상 q99 기준으로 평가했으며 동일 테스트 오탐률 비교가 아니다.

### 정상 데이터에서 확인한 공정 보정 차이

Combined 하락 이후 테스트 라벨을 사용하지 않고 정상 calibration의 전이 점수만 진단했다. 현재는 모든 상태의 `-log P(next | previous) + penalty`를 하나의 정상 empirical CDF로 보정한다.

| 이전 상태 | 정상 전이 표본 | self-transition 비율 | raw 점수 중앙값 | 전역 percentile 중앙값 |
|---|---:|---:|---:|---:|
| latent_0 | 264 | 98.11% | 0.021506 | 0.1899 |
| latent_1 | 211 | 98.10% | 0.021684 | 0.5213 |
| latent_2 | 208 | 96.63% | 0.053328 | 0.8115 |
| latent_3 | 16 | 87.50% | 0.126867 | 0.9644 |

정상 self-transition이 대부분인데 상태별 percentile 수준이 크게 다르다. 특히 latent_0/1의 매우 작은 raw 차이도 큰 percentile 차이로 변환된다. 이는 전역 보정이 상태별 분포와 동점 빈도의 차이를 반영하는 관찰이며, Combined 하락의 원인이 입증됐다는 뜻은 아니다. 조건부 보정의 효과는 별도 실험으로 확인해야 한다.

## 결과의 의의

외형 입력을 그대로 두고 phase 정의를 바꾸어 phase-conditioned PCA의 결과를 비교할 수 있는 구현을 확보했다. Visual AUROC +0.0362, AP +0.0404로 이번 R03에서는 관계 상태가 외형 분포를 나누는 데 도움이 될 가능성이 있다. 반면 두 branch의 개별 ranking이 나아져도 고정 결합의 ranking은 나빠졌다. branch 성능과 최종 결합 성능을 구분하고 정상 보정의 상태 의존성을 검증해야 한다는 구체적 근거다.

새로운 파이프라인 구현·구성 요소 비교·실패 원인 진단을 논문 자료로 축적한다. KMeans, 관계 특징, PCA의 결합 자체가 학술적 novelty라는 주장이나 통계적 유의성·다른 장면 일반화 주장은 하지 않는다.

## 보완할 점

- Combined AUROC/AP는 하락했고 이상 프레임 recall은 여전히 10.36%다. 구간 11개 탐지는 프레임 이상 전체 탐지를 뜻하지 않는다.
- 네 군집은 객체 상태 GT가 아니다. 상대 위치뿐 아니라 지게차 절대 위치가 포함되고, 정상 영상의 상대 시간 순으로 cluster 번호를 정렬했다. 작업 공간 위치·진행 순서 proxy일 수 있다.
- bbox 면적 gate는 배경 후보 억제 heuristic이다. 부분·중복 검출, track 전환, 가림 문제와 독립 bbox/관계 주석 부재가 남아 있다. relation_valid는 실제 관계 정확도가 아니다.
- phase를 바꾸면 외형 PCA와 전이 모두 바뀐다. 결과가 관계 descriptor 자체, 군집 분할, 상태 순서 prior 중 무엇에 의존하는지는 아직 분리하지 않았다.
- calibration 4개 영상에서 latent_3 전이는 16개뿐이다. 프레임 간 상관과 정상 영상 분할의 원본 그룹 독립성 미확인 때문에 표본 수를 독립 증거 수로 볼 수 없다. R01/R03은 반복 관찰한 개발 장면이다.
- Stage 00의 다른 장면 11개 시퀀스 라벨 정렬은 계속 미해결이다. FPS가 없어 지연은 원본 프레임 단위다. 이번 실행은 기존 캐시를 재사용했으며 새 VLM/GPU 추출을 하지 않았다. 전체 실행 비용·추론 지연은 측정하지 않았다.

## 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **이전 상태별 공정 점수 보정**: 정상 calibration 전이를 이전 latent state별로 나누어 empirical percentile 계산 | 정상 percentile 중앙값이 상태별 0.190~0.964로 다름 | phase·외형·raw 전이·결합 비율 유지. 정상 support와 fallback 명시. AUROC/AP, 오탐/recall, 구간 미탐·지연 및 정상 상태별 offset 비교 |
| 2 | **공정 결합 기여 분리**: Visual 단독과 고정 결합, 정상 신뢰도에 근거한 결합 후보 비교 | 각 branch의 ranking은 상승했지만 Combined AUROC는 -0.0385 | test 라벨로 가중치 탐색 금지. branch별 점수 상관·오류 중첩과 동일 평가 규칙을 함께 기록 |
| 3 | **잠재 상태·관측 신뢰성 검증**: 정상 상태 대표 관측, bbox/관계 표본 점검 및 순서 안정성 진단 | 네 상태 support는 확보됐으나 semantic GT가 없고 latent_3 calibration은 16개 | 행동 의미를 임의 부여하지 않음. 보조 주석은 학습과 분리하고 관측 가용성·정확도·상태 support를 각각 평가 |

후속 실험 09는 1순위만 적용한다. 나머지는 확정 실험 일정이 아니며 다음 결과에 따라 순위를 갱신한다. [실험 09 계획](EXPERIMENT09_PLAN.md)

## 검증과 재현

27개 테스트가 통과했다. 39개 특징 캐시에서 기존 배열 중 phase 이외 값의 동일성을 확인했고, 17개 테스트 예측의 라벨 일치·점수 유한성, 설정 freeze의 hash, 수치 self/next/cycle 구조 보존을 검사했다. 인과성·테스트 시간/길이 비의존·큰 배경 anchor 제외·누락 history reset을 단위 테스트했다. 이 검증은 의미 상태 정확도를 보증하지 않는다.

실험 07 특징 캐시를 먼저 준비한 후 저장소 루트에서 실행한다.

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=.cache/matplotlib
.venv/bin/python scripts/prepare_relational_phase.py --config configs/experiment08.json --fit-only
.venv/bin/python scripts/prepare_relational_phase.py --config configs/experiment08.json
.venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset --config configs/experiment08.json
.venv/bin/python scripts/plot_experiment.py --experiment 08
.venv/bin/python scripts/compare_runs.py --experiments 07 08
.venv/bin/python scripts/compare_event_delays.py --experiments 07 08
.venv/bin/python scripts/diagnose_process_calibration.py --config configs/experiment08.json
.venv/bin/python -m pytest -q
```

[설정](../configs/experiment08.json) · [지표](../results/experiment08/metrics.json) · [영상별 결과](../results/experiment08/per_sequence.csv) · [관계 진단](../results/experiment08/relation_features.json) · [정상 보정 진단](../results/experiment08/normal_process_calibration_diagnostic.json) · [구간 지연](../results/comparison07_08/events.json) · [검증 기록](../results/experiment08/validation.json) · [사전 계획](EXPERIMENT08_PLAN.md)
