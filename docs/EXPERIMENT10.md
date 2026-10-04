# 실험 10 — 최댓값 결합과 Visual 단독 비교

## 이번 실험 결과

**최댓값 결합은 평균 결합의 상태 유지 구간 탐지 억제를 해소했지만, Visual 단독 대비 공정 점수의 추가 기여는 작았다.** 평균 대비 recall은 2.29%→25.59%, 정상 오탐률은 1.85%→2.62%다. Visual 단독과의 AUROC 차이는 +0.0009이고, 각자의 정상 q99에서 비교하면 이상 구간 2개를 더 놓쳤다.

### 질문과 실험 범위

실험 09에서 평균 결합은 self-transition 구간의 Visual 점수가 최대여도 정상 q99를 넘지 못했다. 두 branch를 고정하고 최종 결합만 바꾸면 이 문제가 해소되고 추가 공정 증거가 유용한지 검증했다.

| 설정 | 최종 점수 | 역할 |
|---|---|---|
| 실험 09 | `0.5 × Visual + 0.5 × Process` | 기존 비교 기준 |
| 실험 10 | `max(Visual, Process)` | 최댓값 결합 |
| 실험 10_visual | `Visual` | 동일 외형 점수의 대조군 |

- R03 정상 FIT 18개 / calibration 4개 / 테스트 17개, seed 42. 평가 12,005프레임 중 이상 5,068프레임이다. 라벨 길이는 모두 일치하며 반복 관찰한 개발 장면이다.
- 실험 09의 특징 캐시 39개를 각 설정으로 복사했다. 객체 관계 phase·CLIP 특징·PCA·전이 모델·이전 상태별 정상 보정·sampling은 모두 유지했다. 상태 체류 시간은 이번 탐지 점수에 넣지 않았다.
- 두 branch는 세 설정에서 동일하다. Visual AUROC/AP는 0.7330/0.6778, Process AUROC/AP는 0.4627/0.4086이다. Visual 대조군에서도 검증을 위해 Process를 계산하지만 최종 점수에는 사용하지 않는다.
- 각 최종 점수의 정상 calibration q99를 따로 정했다. strict `>` 경보, 4프레임 sampling과 이전 관측 점수 유지 방식은 같다. 가중치 탐색이나 테스트 라벨에 따른 threshold 선택은 하지 않았다.
- [10 평가 전 기록](../results/experiment10/pre_evaluation_protocol.json)과 [10_visual 평가 전 기록](../results/experiment10_visual/pre_evaluation_protocol.json)에 설정·특징 hash를 고정했다. 새 VLM/GPU 특징 추출은 없으며 전체 실행 시간·운용 지연은 측정하지 않았다.

### 탐지 결과

| R03 최종 점수 지표 | 09 평균 | 10 최댓값 | 10_visual 외형 단독 |
|---|---:|---:|---:|
| AUROC | 0.7108 | 0.7339 | 0.7330 |
| AP | 0.6125 | 0.6784 | 0.6778 |
| 정상 q99 | 0.957959 | 0.997354 | 0.996533 |
| 테스트 정상 오탐률 | 1.85% | 2.62% | 7.87% |
| 테스트 이상 프레임 recall | 2.29% | 25.59% | 31.53% |
| 경보가 발생한 GT 이상 구간 | 11 / 17 | 11 / 17 | 13 / 17 |
| 미탐 GT 이상 구간 | 6 | 6 | 4 |

![세 가지 결합 비교](../results/comparison09_10_10_visual/comparison.png)

모든 결과는 **각자의 정상 q99**이며 동일 테스트 오탐률 비교가 아니다. 최댓값은 각 프레임의 Visual 점수보다 작아질 수 없지만, 정상 q99도 높아지므로 기존 Visual 경보의 보존까지 보장하지 않는다.

### 상태 유지 구간과 전환 구간

| 원본 프레임 구분 | 정상 / 이상 수 | 09 오탐 / 이상 탐지 | 10 오탐 / 이상 탐지 | Visual 오탐 / 이상 탐지 |
|---|---:|---:|---:|---:|
| 영상 첫 sampled 구간 | 52 / 16 | 0 / 0 | 0 / 4 | 4 / 8 |
| 추정 상태 유지 | 6,729 / 4,884 | 0 / 0 | 146 / 1,213 | 498 / 1,514 |
| 추정 상태 변경 | 156 / 168 | 128 / 116 | 36 / 80 | 44 / 76 |

최댓값 결합은 상태 유지 이상 1,213프레임에서 경보를 낸다. 평균 결합의 구조적 억제는 해소됐지만, Visual 단독보다 이 구간의 탐지는 301프레임 적다. 상태 유지/변경은 관계 군집 기준이며 실제 행동 정답이 아니다.

### Visual 단독 대비 경보 변화와 상단 점수

| 최댓값 결합의 경보 변화 | 정상 프레임 | 이상 프레임 |
|---|---:|---:|
| Visual 단독에 없던 경보 | 16 | 16 |
| Visual 단독에 있었으나 사라진 경보 | 380 | 317 |
| 두 설정 모두 경보 | 166 | 1,281 |

최댓값 q99는 Visual q99보다 0.0008219 높다. 최댓값이 추가한 이상 경보 16프레임보다 임계값 변화로 잃은 이상 경보 317프레임이 많았다. 이 비교는 점수 결합의 직접 효과와 정상 임계값 재보정의 효과를 함께 포함하므로 공정 branch 자체의 탐지 정확도로 해석하지 않는다.

| 상단 진단 | 10 최댓값 | Visual 단독 |
|---|---:|---:|
| 정상 calibration sampled 수 | 703 | 703 |
| 정상 q99와 같은 표본 | 1 | 1 |
| 정상 q99 초과 표본 | 7 | 7 |
| 정상 score=1 표본 | 0 | 0 |
| 테스트 score=1 정상 / 이상 프레임 | 70 / 765 | 66 / 741 |
| 테스트 고유 최종 점수 수 | 517 | 535 |

두 설정 모두 테스트 점수가 q99와 정확히 같은 프레임은 0개다. 따라서 이번 경보 차이는 q99의 동점 처리 차이가 아니다. 다만 정상 reference를 넘는 테스트 점수가 1로 포화되어 ranking의 동점은 존재한다. Process가 Visual보다 큰 테스트 프레임은 248개로 제한적이다.

### 구간 탐지와 지연

09→10에서는 R03_15의 `[620,692)`를 새로 탐지하고 R03_09의 `[620,712)`를 놓쳤다. 공통 탐지 10개 구간의 지연 변화 중앙값은 -64원본 프레임이다. 탐지 구간만의 지연 중앙값 150→52는 대상 집합이 다르므로 전체 지연 개선량으로 해석하지 않는다.

Visual→10에서는 새로 탐지한 구간 없이 R03_13의 `[428,530)`, R03_14의 `[363,417)`를 추가로 놓쳤다. 공통 탐지 11개 구간의 지연 변화 중앙값은 +4프레임이다. Visual/10의 탐지 구간만의 지연 중앙값은 33/52프레임이다.

각 설정은 미탐을 분모에서 빼지 않고, GT 구간에 경보 하나라도 겹치는지로 구간 탐지를 센다. point adjustment는 하지 않는다. 10과 Visual은 각각 이상 시작 전 활성 경보 1개가 있고, 왼쪽 경계에서 시작한 GT 구간은 4개다. 12프레임 이하 GT 구간은 없어 짧은 이상 보존은 검증하지 못했다.

## 결과의 의의

결합 방식만 분리한 비교로 평균 결합의 상태 유지 구간 억제가 해소됨을 확인했다. 동시에 `max`의 점수 단조성이 재보정된 임계값의 경보 단조성을 보장하지 않는다는 구현상 중요한 차이를 측정했다.

최댓값과 Visual 단독의 AUROC/AP 차이는 각각 +0.000914/+0.000587에 불과하고, 공정 추가로 새 GT 이상 구간을 탐지하지 못했다. 현재 공정 전이 신호의 한계를 인정하고 새로운 시간 정보를 검증할 근거다. 최댓값 결합 자체의 학술적 novelty, 작은 차이의 유의성, 전체 IPAD 일반화는 주장하지 않는다.

## 보완할 점

- 최댓값 결합도 이상 프레임 약 74%와 GT 구간 6개를 놓친다. Visual 대비 낮은 오탐률에는 높은 정상 threshold의 영향이 포함돼 공정 판별력의 개선이라고 단정할 수 없다.
- 정상 calibration 703개 sampled 관측은 영상 4개에서 나온 상관된 표본이다. 매우 작은 threshold 차이에 경보가 크게 달라져 영상별 보정 안정성 진단이 필요하다.
- 기존 Process는 같은 상태가 오래 지속돼도 동일한 self-transition 점수를 반복한다. 상태 체류 시간·상태 내부 관계 변화는 아직 표현하지 않는다.
- 객체 관계 군집은 의미 상태 정답이 아니다. bbox 품질, phase jitter, 가림·검출 실패가 전이와 향후 체류 시간 모델 모두에 영향을 줄 수 있다.
- R01/R03은 개발 장면이며 반복 관찰에 따른 선택 편향이 있다. 다른 장면의 라벨 정렬 미해결 11개 시퀀스와 FPS 부재도 그대로다. 독립 평가와 논문 주제는 충분한 근거를 확보한 뒤 정해야 한다.

### 다음 개선을 위한 정상 FIT 진단

이번 결과를 확인한 뒤 **정상 FIT만** 사용해 연속 관계 상태의 길이를 조사했다. 영상 처음/끝에 걸린 구간과 관계 누락이 있는 구간을 제외한 내부 완결 구간의 진단이며, 아직 이상 점수에는 사용하지 않았다.

| 상태 | 전체 연속 구간 | 영상 시작 / 끝 경계 구간 | 유효 내부 완결 구간 | 완결 길이 최소 / 중앙 / 최대 (프레임) |
|---|---:|---:|---:|---:|
| latent_0 | 40 | 18 / 18 | 4 | 4 / 6 / 8 |
| latent_1 | 18 | 0 / 0 | 17 | 192 / 216 / 244 |
| latent_2 | 36 | 0 / 0 | 36 | 4 / 54 / 244 |
| latent_3 | 25 | 0 / 0 | 25 | 4 / 32 / 108 |

latent_0의 짧은 내부 구간 4개로 시작·종료의 긴 정상 구간을 대표할 수 없다. 따라서 상태별 체류 시간을 무조건 적합하거나 pooled 기준을 복사하면 부당한 경보를 만들 수 있다. 위 표는 진입/종료 인접 관측까지 검사하는 후속 엄격 규칙 적용 전의 support 진단이다.

## 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **관측된 진입 이후의 정상 체류 시간 신호**: 충분한 완결 정상 구간이 있는 상태에서 경과 시간의 긴 꼬리 점수 추가 | 전이 branch의 추가 ranking 이득이 작고 새로운 이상 구간도 없음. 상태 1/2/3은 정상 완결 구간 support 존재 | 상태 0·경계/누락·진입 미관측은 abstain. 정상 진단 후 명세 고정. 지속 구간 탐지·전환 오탐·관측률·지연 함께 검증 |
| 2 | **영상 단위 정상 보정 안정성 검증**: calibration 영상별 상단 분포와 임계값 민감도 분석 | q99 차이 0.0008219에 Visual 경보 697프레임 소실 | 테스트 성능으로 threshold 선택 금지. 프레임 상관과 정상 영상 수를 명시하고 ranking과 운용점 분리 |
| 3 | **관계 상태의 의미·관측 품질 검증**: 정상 대표 구간과 짧은 상태 전환을 점검 | latent_0 내부 완결 구간은 4~8프레임뿐, 의미 GT 없음 | 검출 실패·군집 jitter·실제 전환을 구분. 보조 주석은 학습과 분리하고 상태 정확도를 가용성으로 대체하지 않음 |

다음 실험 11은 1순위를 선택한다. 나머지는 고정된 일정이 아니며 다음 결과에 따라 갱신한다. [실험 11 계획](EXPERIMENT11_PLAN.md)

## 검증과 재현

31개 테스트가 통과했다. 각 설정에서 기존 정상 모델의 threshold 이외 배열이 모두 동일하고, 39개 특징 캐시의 hash가 원본과 일치함을 확인했다. 테스트 예측 17개에서 Visual·Process·객체 점수·phase·라벨이 모두 보존됐고 최종 점수만 바뀌었다. 저장된 정상 점수로 각 q99를 다시 계산해 일치함을 검사했다. 그래프의 표시와 보고서 수치·링크를 확인했다.

실험 09 캐시를 준비한 뒤 저장소 루트에서 실행한다.

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=.cache/matplotlib
for n in 10 10_visual; do
  mkdir -p artifacts/experiment${n}/features
  cp -a artifacts/experiment09/features/. artifacts/experiment${n}/features/
  .venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset --config configs/experiment${n}.json
  .venv/bin/python scripts/diagnose_fusion.py --experiment ${n}
  .venv/bin/python scripts/plot_experiment.py --experiment ${n}
done
.venv/bin/python scripts/validate_fusion_experiment.py
.venv/bin/python scripts/compare_runs.py --experiments 09 10 10_visual
.venv/bin/python scripts/compare_event_delays.py --experiments 09 10
.venv/bin/python scripts/compare_event_delays.py --experiments 10_visual 10
.venv/bin/python scripts/diagnose_dwell_support.py --experiment 10
.venv/bin/python -m pytest -q
```

검증의 cache hash는 원본 실행 캐시 기준이다. 다른 환경에서 재추출한 경우 원본 평가 전 기록을 덮어쓰지 말고 새 provenance를 기록해야 한다.

[10 설정](../configs/experiment10.json) · [Visual 설정](../configs/experiment10_visual.json) · [10 지표](../results/experiment10/metrics.json) · [Visual 지표](../results/experiment10_visual/metrics.json) · [10 영상별 결과](../results/experiment10/per_sequence.csv) · [Visual 영상별 결과](../results/experiment10_visual/per_sequence.csv) · [결합 비교 진단](../results/comparison09_10_10_visual/fusion_diagnostic.json) · [평균 대비 구간 지연](../results/comparison09_10/events.json) · [Visual 대비 구간 지연](../results/comparison10_visual_10/events.json) · [정상 체류 support](../results/experiment10/normal_dwell_support.json) · [검증](../results/experiment10/validation.json) · [사전 계획](EXPERIMENT10_PLAN.md)
