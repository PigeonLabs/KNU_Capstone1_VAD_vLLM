# 실험 09 — 이전 상태별 공정 점수 보정

## 이번 실험 결과

**조건부 보정으로 Combined AUROC는 0.6361→0.7108로 상승했지만, 정상 q99 기준 이상 프레임 recall은 10.36%→2.29%로 하락했다.** 정상 상태별 점수 차이를 줄이는 것과 이상 신호를 잘 표현하는 것은 별개였다. 평균 결합의 임계값에서 상태 유지 구간의 외형 이상 경보가 구조적으로 억제되는 문제를 확인했다.

### 질문·변경·고정 조건

실험 08 정상 calibration에서 이전 상태별 공정 percentile 중앙값은 0.190~0.964였다. 기존 전역 보정 대신 이전 상태별 정상 empirical CDF를 사용하면 이 차이와 최종 결합의 약점을 줄일 수 있는지 검증했다.

- R03 정상 FIT 18개 / calibration 4개 / 테스트 17개 영상, seed 42. 평가 12,005프레임 중 이상 5,068프레임이다. R03은 라벨 길이가 모두 일치하고, 반복 관찰한 개발 장면이다.
- 실험 08의 39개 특징 캐시를 바이트 단위로 복사했다. 객체 관계 phase, detector·tracking·CLIP 특징, PCA, 전이 확률과 self/next/cycle penalty, Visual 보정, 4프레임 sampling을 유지했다.
- 정상 calibration의 각 영상에서 첫 관측을 제외하고 전이 raw 점수를 **이전 상태**별로 모았다. support가 10개 이상이면 해당 상태의 midrank percentile을 사용하고, 부족하거나 이전 상태가 없으면 기존 전역 reference로 fallback한다. 첫 관측은 상태별 reference에는 넣지 않지만 기존 전역 reference에는 그대로 남는다.
- `Combined = 0.5 × Visual + 0.5 × Process`를 유지했다. 바뀐 점수의 정상 calibration에서 최종 q99를 다시 정했다. 경보는 strict `>`이며 원본 프레임 점수는 이전 sampled 값을 유지한다.
- 테스트 라벨로 가중치·support 기준·threshold를 선택하지 않았다. [평가 전 설정과 캐시 hash](../results/experiment09/pre_evaluation_protocol.json)를 저장했다. 새 VLM 호출이나 GPU 특징 추출은 없었다.

조건부 reference의 정상 전이 표본은 상태 0/1/2/3별 264/211/208/16개다. 테스트 sampled 3,007개 중 2,990개(99.43%)에 조건부 보정을 적용했다. fallback은 각 영상의 첫 관측 17개이며 support 부족 fallback은 0개다.

### 탐지 결과

| R03 지표 | 실험 08: 전역 보정 | 실험 09: 이전 상태별 보정 |
|---|---:|---:|
| Visual AUROC | 0.7330 | 0.7330 |
| Visual AP | 0.6778 | 0.6778 |
| Process AUROC | 0.5520 | 0.4627 |
| Process AP | 0.4693 | 0.4086 |
| Combined AUROC | 0.6361 | 0.7108 |
| Combined AP | 0.5496 | 0.6125 |
| 정상 q99 threshold | 0.966081 | 0.957959 |
| 테스트 정상 오탐률 | 3.36% | 1.85% |
| 테스트 이상 프레임 recall | 10.36% | 2.29% |
| 경보가 발생한 GT 이상 구간 | 11 / 17 | 11 / 17 |
| 미탐 GT 이상 구간 | 6 | 6 |

![실험 08–09 비교](../results/comparison08_09/comparison.png)

두 실험이 탐지한 11개 구간은 동일하다. 공통 탐지 구간의 첫 경보 지연 변화 중앙값은 0프레임, 탐지 구간만의 지연 중앙값은 모두 150프레임이다. R03_02는 184프레임 빨라졌고 R03_09/10은 각각 36/135프레임 늦어졌다. 실험 08의 시작 전 활성 경보 1개는 실험 09에서 없어졌다. 왼쪽 영상 경계에서 시작한 GT 구간 4개와 미탐 6개는 그대로다. 12프레임 이하 GT 이상 구간은 0개다.

구간 탐지는 GT 연속 이상 구간에 경보가 하나라도 겹치는지이며 point adjustment를 하지 않는다. 각자의 정상 q99 기준이므로 동일 테스트 오탐률 비교가 아니다. 구간 탐지 수가 유지돼도 이상 프레임 recall의 하락을 숨기지 않는다.

### 정상 상태별 보정 진단

| 이전 상태 | 정상 전이 표본 | 전역 percentile 중앙값 | 조건부 percentile 중앙값 |
|---|---:|---:|---:|
| latent_0 | 264 | 0.1899 | 0.4905 |
| latent_1 | 211 | 0.5213 | 0.4905 |
| latent_2 | 208 | 0.8115 | 0.4832 |
| latent_3 | 16 | 0.9644 | 0.4375 |

정상 상태별 중앙값 범위는 0.190~0.964에서 0.438~0.491로 줄었다. 이는 **보정에 사용한 동일 정상 calibration**에서의 기술 통계이므로 독립 정상 영상에 대한 일반화 증거가 아니다. 이산 전이의 동점과 상태별 전이 비율 때문에 완전히 같은 분포가 되지는 않는다. Process AUROC는 오히려 하락했으므로 상태별 차이를 없애는 것이 더 나은 이상 구별을 보장하지 않는다.

### 결합 실패의 구체적 근거

각 상태의 self-transition Process 점수는 0.4375~0.4905다. Visual percentile은 최대 1이므로 평균 결합의 상한은 0.7188~0.7453이며, 정상 q99 0.9580보다 낮다. 따라서 **현재 설정에서 추정 상태가 유지되는 sampled 구간은 외형 이상 점수가 최대여도 경보를 낼 수 없다.** 이 상한은 정상 데이터의 보정값과 결합 수식으로 계산된다.

| 원본 프레임 구분 | 정상 프레임 | 이상 프레임 | 정상 오탐 | 이상 탐지 |
|---|---:|---:|---:|---:|
| 각 영상 첫 sampled 구간 | 52 | 16 | 0 | 0 |
| 추정 상태 유지 | 6,729 | 4,884 | 0 | 0 |
| 추정 상태 변경 | 156 | 168 | 128 | 116 |

테스트 라벨은 평가 후 위 구분의 오류 규모를 기술하는 데만 사용했다. 상태 유지/변경은 관계 군집의 변화이며 실제 행동 상태의 정답이 아니다. 상태 유지 이상 4,884프레임(전체 이상의 96.37%)에서 경보가 없었다. 모든 경보가 상태 변경 구간에 집중돼, 이 구간의 정상 156프레임 중 128프레임도 오탐이었다. 낮은 전체 오탐률이 공정 전환 구간의 안전한 판별을 뜻하지 않는다.

## 결과의 의의

외형 특징·phase·PCA·raw 전이를 고정하고 보정만 바꾸어 최종 결합의 변화를 분리했다. Combined AUROC +0.0747, AP +0.0629는 정상 보정 방식이 파이프라인 결과에 큰 영향을 줄 수 있음을 보여준다. 그러나 Process ranking은 악화됐고 Combined는 Visual 단독 AUROC/AP 0.7330/0.6778보다 낮다. 공정 이해 자체가 개선됐다는 해석은 지지하지 않는다.

논문 자료로서의 가치는 정상 상태별 보정과 평균 결합의 역할·실패 조건을 명시하고 재현 가능한 비교를 만든 데 있다. 조건부 CDF 자체의 novelty, 통계적 유의성, 전체 IPAD 일반화는 주장하지 않는다.

## 보완할 점

- 이상 프레임 recall은 2.29%에 그친다. AUROC 개선만으로 현재 운용 임계값의 탐지 성능이 좋아졌다고 결론 내릴 수 없다.
- 상태가 유지되면 기존 전이 모델은 매 관측마다 동일한 self-transition을 본다. 체류 시간·장기 정지·상태 내 객체 관계 변화를 직접 모델링하지 않는다.
- 평균 결합은 정상적인 공정 점수 때문에 높은 외형 점수를 낮춘다. 단, 점수를 낮추지 않는 다른 결합도 정상 q99가 다시 바뀌므로 탐지 recall을 자동 보장하지 않는다.
- latent_3 calibration은 16개 전이뿐이고 프레임 상관이 크다. 상태별 reference가 독립 표본 16개인 것은 아니다. calibration 영상이 4개인 데 따른 안정성은 추가 검증이 필요하다.
- 관계 phase의 의미 GT·bbox 정확도·원본 그룹 독립성은 미확인이다. Stage 00의 다른 장면 11개 시퀀스 라벨 정렬도 여전히 미해결이다. FPS가 없어 지연은 원본 프레임 단위이며 이번 실험의 실행 시간·운용 지연은 측정하지 않았다.

## 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **외형 점수를 낮추지 않는 결합**: 고정 평균을 `max(Visual, Process)`로 교체하고 Visual 단독을 대조군으로 추가 | self-transition 결합 상한이 q99보다 낮아 이상 4,884프레임에서 경보 불가 | 두 branch와 보정 고정. 각 정상 q99로 평가하며 상태 유지 탐지·전환 오탐·미탐/지연을 함께 확인. 가중치 탐색 없음 |
| 2 | **체류 시간 기반 공정 신호**: 정상 상태의 머무름 길이로 전이 빈도 모델의 시간적 한계 보완 | Process AUROC 0.4627, 상태 유지 중에는 동일한 점수 반복 | 원본 프레임 단위, 인과적 경과 시간 사용. 정상 대기와 실제 정지 구분 및 구간 경계 censoring을 먼저 진단 |
| 3 | **정상 보정·상태 관측의 안정성 검증**: 영상 단위 support와 정상 대표 관측 점검 | calibration 4개 영상, 희소 상태 전이 16개, 의미 GT 없음 | 시간 상관을 고려한 영상 단위 검증. 보조 주석을 학습과 분리하고 coverage를 정확도로 해석하지 않음 |

후속 실험 10은 1순위만 선택한다. 나머지는 결과에 따라 재정렬할 후보이며 확정된 전체 실험 일정이 아니다. [실험 10 계획](EXPERIMENT10_PLAN.md)

## 검증과 재현

29개 테스트를 통과했다. 이전 상태 기준 적용, 영상 첫 관측 제외, support 부족 fallback, 미래 관측 비의존, Visual/raw 전이 보존 및 변경된 점수에 대한 정상 q99를 검사했다. 실제 캐시 39개와 예측 17개를 검증해 특징·phase·Visual·객체 점수·라벨의 동일성을 확인했다. 정상 모델의 기존 모든 배열은 threshold를 제외하고 동일하다. 상태별 reference를 정상 calibration에서 재구성하고 이전/새 점수를 각각 전역/조건부 CDF로 다시 계산해 저장값과 일치함을 확인했다.

실험 08 캐시를 준비한 뒤 저장소 루트에서 실행한다.

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=.cache/matplotlib
mkdir -p artifacts/experiment09/features
cp -a artifacts/experiment08/features/. artifacts/experiment09/features/
.venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset --config configs/experiment09.json
.venv/bin/python scripts/diagnose_process_calibration.py --config configs/experiment09.json
.venv/bin/python scripts/diagnose_fusion.py --experiment 09
.venv/bin/python scripts/validate_process_calibration.py --baseline 08 --experiment 09
.venv/bin/python scripts/plot_experiment.py --experiment 09
.venv/bin/python scripts/compare_runs.py --experiments 08 09
.venv/bin/python scripts/compare_event_delays.py --experiments 08 09
.venv/bin/python -m pytest -q
```

검증 스크립트의 캐시 hash 비교는 이 실행의 원본 실험 08 캐시를 기준으로 한다. 다른 환경에서 특징을 재추출하면 새 provenance를 별도로 기록해야 하며 원본 평가 전 기록을 덮어쓰지 않는다.

[설정](../configs/experiment09.json) · [지표](../results/experiment09/metrics.json) · [영상별 결과](../results/experiment09/per_sequence.csv) · [정상 보정 진단](../results/experiment09/normal_process_calibration_diagnostic.json) · [결합 진단](../results/experiment09/fusion_diagnostic.json) · [구간별 지연](../results/comparison08_09/events.json) · [검증](../results/experiment09/validation.json) · [사전 계획](EXPERIMENT09_PLAN.md)
