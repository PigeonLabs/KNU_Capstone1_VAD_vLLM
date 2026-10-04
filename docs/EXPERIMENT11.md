# 실험 11 — 관측된 진입 이후 정상 체류 시간

## 이번 실험 결과

**체류 시간 추가는 이번 R03에서 최종 성능을 악화시켰다.** Combined AUROC는 0.7339→0.7005, AP는 0.6784→0.6055로 하락했다. 정상 오탐 260프레임과 이상 탐지 32프레임이 추가됐으며 새 GT 이상 구간은 탐지하지 못했다.

### 질문·변경·고정 조건

실험 10에서 전이 점수의 추가 기여가 작았고, self-transition은 상태가 오래 지속돼도 같은 점수를 반복했다. 관측된 진입 이후 경과 시간으로 지속 이상 신호를 추가할 수 있는지 검증했다.

- R03 정상 FIT 18개 / calibration 4개 / 테스트 17개, seed 42. 평가 12,005프레임 중 이상 5,068프레임이며 R03 라벨 길이는 모두 일치한다. 반복 관찰한 개발 장면이다.
- 실험 10의 특징·객체 관계 phase·PCA·기존 전이 확률과 이전 상태별 보정을 유지했다. 체류 점수만 추가하고 유효할 때 `Process = max(기존 전이, 체류)`로 합쳤다. 무효일 때 기존 전이를 그대로 사용한다. 최종 `max(Visual, Process)`와 정상 calibration q99 규칙은 유지했다.
- 정상 FIT의 내부 완결 구간 중 진입 전 관측부터 종료 관측까지 모두 relation_valid인 구간만 적합했다. 최소 완결 구간 10개가 필요한데 상태 0/1/2/3의 support는 4/17/36/25개이므로 1/2/3만 사용한다. pooled fallback은 없다.
- 추론 시 양쪽 관계 관측이 유효한 상태 변화에서 진입을 확정한다. 경과 시간은 현재 원본 프레임 인덱스와 진입 인덱스의 차이다. 영상 첫 구간은 진입 미관측이며, 관계 누락 후에는 새 유효 진입까지 abstain한다. 미래 종료 시점이나 테스트 영상 길이는 사용하지 않는다.
- 정상 완결 길이 D, 현재 경과 시간 a에 대해 `raw = -log((1 + count(D >= a)) / (1 + n))`를 사용했다. 정상 calibration에서 **유효한 raw 435개만** 전역 midrank CDF reference로 사용한다. 최소 calibration 표본은 10개다. 짧은/생략 단계 이상은 이 체류 신호의 대상이 아니다.
- 정상 진단 후 [평가 전 설정·코드·특징 hash](../results/experiment11/pre_evaluation_protocol.json)를 고정했다. 테스트 라벨로 support·가중치·threshold를 선택하지 않았다. 4프레임 sampling과 이전 sampled 점수 유지, strict `>` 경보는 같다. 새 VLM/GPU 특징 추출은 없었으며 실행 비용·운용 지연은 측정하지 않았다.

### 체류 관측 가용성

| 구분 | 유효 / 전체 sampled 관측 | 유효 비율 |
|---|---:|---:|
| 정상 FIT | 1,936 / 3,142 | 61.62% |
| 정상 calibration | 435 / 703 | 61.88% |

테스트 원본 프레임에서는 7,014/12,005=58.43%가 유효하다. 유효 구간의 정상/이상은 3,912/3,102프레임이다. 무효 구간은 진입 미관측 4,096프레임, 지원 부족 상태 895프레임이고 테스트 관계 누락은 0프레임이다. 이번 테스트에서 무효 구간은 모두 상태 0이다. FIT의 관계 누락 6개 sampled 관측과 누락 이후 진입 미확정 구간도 별도 처리했다.

체류 단독의 유효 구간 AUROC/AP는 0.5050/0.4253이다. 이는 7,014프레임만의 지표로, 전체 12,005프레임의 다른 branch 지표와 직접 비교하지 않는다. 전체 탐지 평가는 무효 구간도 포함한다.

### 탐지 결과

| R03 지표 | 실험 10 | 실험 11 |
|---|---:|---:|
| Visual AUROC | 0.7330 | 0.7330 |
| Visual AP | 0.6778 | 0.6778 |
| Process AUROC | 0.4627 | 0.5230 |
| Process AP | 0.4086 | 0.4240 |
| Combined AUROC | 0.7339 | 0.7005 |
| Combined AP | 0.6784 | 0.6055 |
| 정상 q99 | 0.997354 | 0.997701 |
| 테스트 정상 오탐률 | 2.62% | 6.37% |
| 테스트 이상 프레임 recall | 25.59% | 26.22% |
| 경보가 발생한 GT 이상 구간 | 11 / 17 | 11 / 17 |
| 미탐 GT 이상 구간 | 6 | 6 |

![실험 10–11 비교](../results/comparison10_11/comparison.png)

각 모델의 정상 q99 기준이며 동일 테스트 오탐률 비교가 아니다. 같은 11개 GT 구간을 탐지했고 새 탐지·탐지 손실 구간은 없다. 공통 첫 경보 지연 변화 중앙값은 0프레임, 탐지 구간만의 지연 중앙값은 모두 52프레임이다. 두 모델 모두 시작 전 활성 경보 1개, 왼쪽 경계에서 시작하는 GT 구간 4개다. 12프레임 이하 GT 구간은 0개다. 미탐은 분모에 남기고 point adjustment는 하지 않았다.

### 실패가 집중된 구간

| 추정 상태 | 정상 / 이상 프레임 | 추가 정상 오탐 | 추가 이상 탐지 |
|---|---:|---:|---:|
| latent_0 | 3,025 / 1,966 | 0 | 0 |
| latent_1 | 1,976 / 1,228 | 0 | 0 |
| latent_2 | 1,770 / 1,382 | 224 | 0 |
| latent_3 | 166 / 492 | 36 | 32 |

체류 점수가 기존 최종 점수를 높인 프레임은 정상 788개, 이상 280개다. 그중 기존에 없던 경보는 정상 260개, 이상 32개였다. 기존 경보 손실은 없었지만 이는 이번 결과일 뿐, 재보정된 threshold의 경보 보존을 일반적으로 보장하지 않는다. 체류 percentile이 1로 포화된 유효 테스트 프레임은 정상 284개, 이상 136개였다.

무효 체류 구간에서는 기존 점수가 정확히 보존됐고 이번에는 경보도 바뀌지 않았다. 상태 2의 추가 오탐 224개는 모두 `1→2`로 진입한 구간이며, 상태 3의 추가 오탐/탐지는 모두 `2→3` 진입 구간이다. 이 구분은 평가 후 오류 진단이며 실제 행동·방향 정답이 아니다.

## 결과의 의의

경계가 잘린 구간·누락·지원 부족을 명시적으로 제외하는 인과적 체류 시간 모듈을 구현하고, 기존 branch가 보존된 상태에서 그 기여를 측정했다. Process ranking이 조금 높아져도 최종 결합은 악화될 수 있음을 다시 확인했다. 이 모듈을 현재 결과만으로 성능 개선으로 채택할 근거는 없다.

논문 자료로는 정상 상태의 지속 시간까지 확장한 파이프라인 구현과 그 실패 조건을 재현 가능하게 남긴 가치가 있다. 생존 비율이나 체류 모델 자체의 학술적 novelty, 통계적 유의성, 다른 장면 일반화는 주장하지 않는다.

## 보완할 점

- 최종 AUROC/AP가 하락하고 추가 정상 오탐이 추가 이상 탐지보다 훨씬 많다. 단순히 정상보다 오래 지속된다는 이유만으로 이상을 구별하기 어렵다.
- 같은 관계 군집을 다른 경로로 재방문할 수 있는데 현재 모델은 진입 맥락을 합쳐 상태별 길이 하나로 다룬다. 이것이 실패의 원인인지는 아직 입증하지 않았다.
- 정상 FIT에서 경계가 잘린 구간은 제외했다. 이 모델은 censoring을 통계적으로 보정하는 survival estimator가 아니며, 긴 구간을 선택적으로 제외할 가능성이 있다. 상태 0과 영상 시작부터 존재하는 이상에서는 체류 신호를 내지 못한다.
- 정상 calibration은 4개 영상의 상관된 관측 435개다. 상태 3의 calibration 유효 관측도 16개뿐이며 전역 CDF에 다른 상태가 섞인다. 희소 꼬리와 정상 속도 변동의 안정성은 미확인이다.
- 관계 군집의 의미 GT가 없어 검출/군집의 흔들림과 실제 동작 시간을 구분할 수 없다. 객체 bbox 품질, 원본 그룹 독립성, 다른 장면 11개 시퀀스의 라벨 정렬과 FPS 부재도 해결되지 않았다. R01/R03은 개발 장면이다.

### 다음 개선을 위한 정상 진입 맥락 진단

평가 후 정상 FIT/calibration만으로 같은 상태에 들어오는 이전 상태를 구분했다. 아래는 엄격 완결 관측 기준이다.

| 진입 맥락 | FIT 완결 구간 / 영상 수 | FIT 길이 최소 / 중앙 / 최대 | calibration 완결 구간 / 영상 수 |
|---|---:|---:|---:|
| 0→1 | 17 / 17 | 192 / 216 / 244 | 4 / 4 |
| 1→2 | 18 / 18 | 16 / 134 / 244 | 4 / 4 |
| 3→2 | 16 / 10 | 4 / 18 / 112 | 2 / 2 |
| 2→3 | 23 / 11 | 4 / 36 / 108 | 2 / 2 |

상태 2는 진입 경로에 따라 정상 길이 중앙값이 134 대 18프레임으로 다르다. FIT의 0→2, 0→3은 각각 2개뿐이다. 따라서 진입 맥락 조건화는 검증할 근거가 있지만, 지원 부족 구간에서 신호를 끄는 효과와 정상 길이 모델 자체의 효과가 섞일 수 있다. 추가 대조군으로 이 둘을 분리해야 한다. 관측된 정상 길이 차이가 테스트 정상의 긴 체류나 상태 추정 오류까지 해결한다는 보장은 없다.

## 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **진입 맥락별 체류 시간**: 관측된 이전 상태와 현재 상태 쌍으로 정상 길이 분리 | 같은 상태 2의 정상 중앙값이 1→2에서 134, 3→2에서 18프레임 | 최소 support 유지. 동일 유효 마스크의 상태별 길이 대조군을 두어 abstention과 조건화 효과 분리. 오탐·가용성·지연 함께 평가 |
| 2 | **정상 속도·상단 보정 안정성**: 영상별 길이 꼬리와 calibration 대표성 점검 | 추가 오탐 260프레임, 유효 테스트 정상 284프레임에서 체류 percentile=1 | 테스트 threshold 최적화 금지. 정상 영상 단위 검증과 경계 censoring 영향을 구분 |
| 3 | **관계 상태와 실제 동작의 대응 검증**: 대표 정상/오류 구간을 관측 품질 관점에서 점검 | 체류 단독 AUROC 0.5050, 의미/객체 GT 없음 | 군집 재방문·jitter·가림과 실제 지속을 구분. 보조 주석은 학습과 분리하고 정확도를 support로 대체하지 않음 |

다음 실험 12는 1순위만 선택한다. 나머지는 새 결과에 따라 재정렬할 후보이며 전체 실험 일정이 아니다. [실험 12 계획](EXPERIMENT12_PLAN.md)

## 검증과 재현

35개 테스트를 통과했다. 완결 구간의 진입/종료 인접 관측 검사, 영상 경계 제외, 누락 후 진입 재확인, 시간 인과성, 지원 부족 abstention, invalid calibration 제외 및 기존 branch 보존을 검증했다. 실제 특징 캐시 39개와 예측 17개에서 특징·phase·Visual·객체 점수·라벨·기존 전이 점수가 동일함을 확인했다. 기존 정상 모델 배열은 threshold 외 모두 동일하다. 체류 길이/reference를 정상 캐시로 재구성하고 최종 결합·정상 q99를 저장값과 대조했다.

실험 10 캐시를 준비한 후 저장소 루트에서 실행한다.

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=.cache/matplotlib
mkdir -p artifacts/experiment11/features
cp -a artifacts/experiment10/features/. artifacts/experiment11/features/
.venv/bin/python scripts/diagnose_normal_dwell.py --config configs/experiment11.json
.venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset --config configs/experiment11.json
.venv/bin/python scripts/validate_dwell_experiment.py --baseline 10 --experiment 11
.venv/bin/python scripts/compare_runs.py --experiments 10 11
.venv/bin/python scripts/compare_event_delays.py --experiments 10 11
.venv/bin/python scripts/plot_experiment.py --experiment 11
.venv/bin/python scripts/diagnose_dwell_context.py --experiment 11
.venv/bin/python scripts/diagnose_dwell_entries.py --baseline 10 --experiment 11
.venv/bin/python -m pytest -q
```

검증은 원본 실행의 cache/code hash를 기준으로 한다. 다른 환경이나 코드 버전의 재실행은 새 provenance를 기록하고 원본 평가 전 기록을 덮어쓰지 않는다.

[설정](../configs/experiment11.json) · [지표](../results/experiment11/metrics.json) · [영상별 결과](../results/experiment11/per_sequence.csv) · [정상 진단](../results/experiment11/normal_dwell_diagnostic.json) · [체류 오류 진단](../results/experiment11/dwell_diagnostic.json) · [진입별 오류](../results/experiment11/dwell_entry_errors.json) · [정상 진입 맥락](../results/experiment11/normal_entry_context_diagnostic.json) · [구간 지연](../results/comparison10_11/events.json) · [검증](../results/experiment11/validation.json) · [사전 계획](EXPERIMENT11_PLAN.md)
