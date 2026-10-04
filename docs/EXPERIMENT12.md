# 실험 12 — 진입 맥락별 체류 시간과 동일 support 대조

## 이번 실험 결과

**진입 맥락별 체류 모델은 동일 유효 구간 대조군보다 최종 성능이 낮았다.** Combined AUROC는 0.7006→0.6968, AP는 0.6055→0.5953이며 정상 오탐 68프레임과 이상 탐지 24프레임이 추가됐다. 새 GT 이상 구간은 탐지하지 못했다. 정상 길이 분포가 그대로인 맥락도 전역 보정 reference의 변화로 영향을 받는 현상을 확인했다.

### 질문·변경·고정 조건

실험 11에서 같은 상태 2의 정상 길이 중앙값은 1→2 진입 시 134프레임, 3→2 진입 시 18프레임이었다. 진입 맥락으로 정상 길이를 나누는 효과를, 지원 부족 구간에서 체류 신호를 끄는 효과와 분리해 검증했다.

| 설정 | 체류 길이 분포 | 유효 구간 |
|---|---|---|
| 11 | 현재 상태별 pooled 길이 | 상태별 FIT support ≥10 |
| 12_support | 11과 같은 상태별 pooled 길이 | 진입 맥락별 FIT support ≥10 |
| 12 | 진입 이전 상태·현재 상태별 길이 | 12_support와 동일 |

- R03 정상 FIT 18개 / calibration 4개 / 테스트 17개, seed 42. 평가 12,005프레임 중 이상 5,068프레임이며 라벨 길이는 모두 일치한다. R03은 반복 관찰한 개발 장면이다.
- 실험 11의 특징·관계 phase·Visual·기존 전이·정상 상태별 PCA를 유지했다. 진입 이전 상태는 체류 구간 전체에서 고정하고, 관계 누락 시 버린다. 매 프레임의 직전 상태를 진입 맥락으로 잘못 사용하지 않는다.
- 완결 정상 길이는 실험 11과 같은 엄격한 진입/종료 인접 관측 기준으로 수집했다. 지원 맥락은 0→1, 1→2, 2→3, 3→2이고 FIT 완결 구간은 각각 17/18/23/16개다. 지원 부족 맥락·진입 미관측·관계 누락에서는 체류 신호를 내지 않는다.
- raw survival 수식은 유지했다. 각 설정의 유효 정상 raw만 전역 midrank CDF로 보정하고, 체류/전이 max 및 Visual/Process max로 결합했다. 각 최종 점수의 정상 q99를 다시 적합했다. 4프레임 sampling, 이전 관측 점수 유지와 strict `>` 경보는 같다.
- 두 설정 모두 유효 관측은 FIT 1,923/3,142=61.20%, calibration 424/703=60.31%다. 테스트에서는 6,910/12,005=57.56%이며 정상/이상 3,864/3,046프레임이다. 원본 11보다 정상 48·이상 56프레임에서 체류 신호를 더 끈다.
- 정상 진단 후 설정·코드·특징 hash를 고정했다. 테스트 라벨로 support·가중치·threshold를 선택하지 않았다. 새 VLM/GPU 특징 추출은 없었고 실행 비용·운용 지연은 측정하지 않았다.

### 탐지 결과

| R03 지표 | 11 상태별 길이 | 12_support 동일 support | 12 진입별 길이 |
|---|---:|---:|---:|
| Visual AUROC | 0.7330 | 0.7330 | 0.7330 |
| Visual AP | 0.6778 | 0.6778 | 0.6778 |
| Process AUROC | 0.5230 | 0.5253 | 0.5278 |
| Process AP | 0.4240 | 0.4248 | 0.4285 |
| Combined AUROC | 0.7005 | 0.7006 | 0.6968 |
| Combined AP | 0.6055 | 0.6055 | 0.5953 |
| 정상 q99 | 0.997701 | 0.997642 | 0.997642 |
| 테스트 정상 오탐률 | 6.37% | 6.37% | 7.35% |
| 테스트 이상 프레임 recall | 26.22% | 26.22% | 26.70% |
| 경보가 발생한 GT 이상 구간 | 11 / 17 | 11 / 17 | 11 / 17 |
| 미탐 GT 이상 구간 | 6 | 6 | 6 |

![실험 12 대조 비교](../results/comparison11_12_support_12/comparison.png)

12와 12_support의 q99는 이번 실행에서 정확히 같은 값 0.9976415094339622였다. 11과의 threshold는 다르다. 모두 정상 데이터에서 정했으며 동일 테스트 오탐률을 맞춘 비교가 아니다.

체류 단독의 **동일 유효 6,910프레임** AUROC/AP는 대조군 0.5082/0.4246, 진입별 0.5172/0.4316이다. 원본 11의 체류 단독 지표는 유효 프레임 집합이 달라 직접 같은 범위의 비교로 취급하지 않는다. 전체 지표는 체류 무효 구간도 포함한다.

### 지원 마스크 효과와 길이 조건화 효과

11→12_support에서는 점수와 reference가 조금 바뀌었지만 경보가 추가되거나 사라진 프레임은 없었다. 12_support→12에서는 정상 오탐 68개, 이상 탐지 24개가 추가됐고 기존 경보 손실은 없었다. 따라서 이번 추가 경보는 서로 다른 유효 구간을 비교한 결과가 아니다.

| 진입 맥락 | 유효 정상 / 이상 프레임 | 진입별 모델의 추가 정상 오탐 | 추가 이상 탐지 |
|---|---:|---:|---:|
| 0→1 | 1,976 / 1,180 | 48 | 0 |
| 1→2 | 1,706 / 1,230 | 0 | 0 |
| 2→3 | 166 / 484 | 20 | 24 |
| 3→2 | 16 / 152 | 0 | 0 |

실험 11에서 상태 2에 추가됐던 정상 오탐 문제는 해결되지 않았다. 진입별 모델은 오히려 0→1과 2→3에서 경보를 늘렸다. 체류 percentile=1인 유효 테스트 정상/이상 프레임도 대조군 284/136에서 진입별 360/184로 증가했다.

세 설정은 동일한 11개 GT 구간을 탐지했다. 두 비교 모두 공통 첫 경보 지연 변화 중앙값은 0프레임, 탐지 구간만의 지연 중앙값은 52프레임이다. 각 설정에 시작 전 활성 경보 1개가 있고 왼쪽 경계에서 시작한 GT 구간은 4개다. 12프레임 이하 GT 구간은 0개다. 미탐은 분모에 포함하고 point adjustment를 하지 않았다.

### 다른 맥락이 바뀌어도 영향을 받는 전역 보정

0→1의 정상 FIT 길이 17개는 두 설정에서 정확히 같다. 정상 calibration의 해당 맥락 211개 raw 점수도 동일하다. 그러나 전체 맥락을 합친 reference가 바뀌어 같은 raw가 다른 percentile이 된다.

| 0→1 정상 calibration 진단 | 12_support | 12 |
|---|---:|---:|
| raw 최소 / 중앙 / 최대 | 0 / 0 / 0.587787 | 동일 |
| percentile 최소 / 중앙 / 최대 | 0.2500 / 0.2500 / 0.6427 | 0.2642 / 0.2642 / 0.8550 |

기존 Visual·전이·진입 맥락·길이 분포와 두 threshold가 같은데도 0→1의 추가 정상 오탐 48개가 발생했다. 이 설정에서는 다른 맥락의 raw 변화가 공유 CDF를 통해 전달된 영향으로 설명할 수 있다. 다만 이 보정 진단은 보정에 사용한 정상 영상을 재사용하므로 독립 정상 영상에서의 안정성을 입증하지 않는다.

## 결과의 의의

동일 유효 구간 대조군으로 abstention과 체류 조건화의 효과를 분리했다. 정상 길이의 맥락 차이가 있다는 사실만으로 최종 이상탐지 개선을 기대할 수 없다는 결과다. 또한 맥락별 길이 분포를 바꾸면 바꾸지 않은 맥락의 보정 점수도 영향을 받을 수 있음을 검증했다.

이 모듈의 성능 개선이나 학술적 novelty는 현재 결과가 지지하지 않는다. 논문 자료로는 정상 기반 모델·보정·결합의 역할을 구분한 재현 가능한 구현과 반례가 축적됐다. 작은 Process/체류 단독 지표 차이를 유의성이나 일반화로 주장하지 않는다.

## 보완할 점

- 최종 ranking은 낮아졌고 새 GT 이상 구간을 탐지하지 못했다. 기존 긴 정상 체류의 오탐 문제도 남았다. 더 많은 조건을 넣는 방향을 성능 개선으로 간주할 수 없다.
- calibration 424개 sampled 관측은 영상 4개에서 나온 상관된 표본이다. 진입별 조건화와 공유 CDF의 간섭을 진단했지만, 그 reference와 threshold가 다른 정상 영상에도 안정적인지 아직 직접 검증하지 않았다.
- 지원 부족 맥락은 기존 전이를 유지하므로 일부 체류 이상을 표현할 수 없다. 12의 체류 무효 프레임은 진입 미관측 4,096개와 지원 부족 맥락 999개다. 무효를 정상으로 판정한 것은 아니다.
- 진입 맥락은 관계 군집의 이전 상태일 뿐 실제 공정 의미·방향 GT가 아니다. 군집 jitter, 검출 오류, 정상 속도 변동과 경계가 잘린 긴 구간의 제외가 길이 모델을 왜곡할 수 있다.
- R01/R03은 반복 개선에 사용한 개발 장면이다. 원본 그룹 독립성, 객체/phase GT, 다른 장면 11개 시퀀스의 라벨 정렬, FPS 부재는 해결되지 않았다.

## 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **정상 영상 단위 보정 검증 추가**: calibration 4개 영상을 하나씩 보정에서 제외해 별도로 평가 | 동일한 0→1 raw의 percentile도 공유 CDF에 따라 달라짐. 정상 오탐이 연속 실험에서 증가 | FIT 18개는 고정, 3개 정상 영상으로 모든 reference/q99를 적합하고 제외한 1개에서 오탐 평가. test 라벨·모델 선택 없이 영상별 안정성 확인 |
| 2 | **맥락 간 보정 간섭 완화**: 충분한 정상 support에서 맥락별 CDF 또는 계층적 보정 후보 검증 | 길이 분포가 그대로인 0→1에서 추가 정상 오탐 48개 | 우선 1순위로 실제 정상 일반화와 support 부족을 확인. 동일 마스크·raw 점수 대조로 보정만 분리하고 sparse context의 불안정성 기록 |
| 3 | **관계 상태의 의미·관측 재검증**: 정상 재방문과 오류 구간의 객체 관계를 점검 | 진입별 체류 단독 AUROC 0.5172, 상태 2의 기존 오탐 미해결 | 실제 대기/방향 변화와 군집·검출 오류 구분. 보조 주석은 학습과 분리하고 단순 관측률을 정확도로 취급하지 않음 |

다음 실험 13은 1순위를 수행한다. 새로운 체류/보정 함수를 더 넣기 전에 정상 보정 검증 절차를 파이프라인에 추가한다. 후보 전체를 확정 일정으로 취급하지 않는다. [실험 13 계획](EXPERIMENT13_PLAN.md)

## 검증과 재현

38개 테스트를 통과했다. 진입 맥락의 구간 내 유지·누락 시 reset·미래 비의존, 조건별 raw 차이, 동일 support 마스크, 기존 Visual/전이 보존을 검사했다. 실제 각 설정의 특징 캐시 39개와 예측 17개에서 기존 특징·phase·Visual·전이·객체 점수·라벨이 일치했다. 두 설정의 calibration/test 유효 마스크가 같고, 대조군의 유효 raw가 원본 상태별 raw와 같음을 확인했다. 정상 길이/reference·최종 결합·q99를 재구성하고 설정·코드 hash를 확인했다.

실험 11 캐시를 준비한 뒤 저장소 루트에서 실행한다.

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=.cache/matplotlib
for n in 12 12_support; do
  mkdir -p artifacts/experiment${n}/features
  cp -a artifacts/experiment11/features/. artifacts/experiment${n}/features/
  .venv/bin/python scripts/diagnose_normal_dwell.py --config configs/experiment${n}.json
  .venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset --config configs/experiment${n}.json
  .venv/bin/python scripts/plot_experiment.py --experiment ${n}
done
.venv/bin/python scripts/validate_context_dwell.py
.venv/bin/python scripts/diagnose_context_calibration.py
.venv/bin/python scripts/compare_runs.py --experiments 11 12_support 12
.venv/bin/python scripts/compare_event_delays.py --experiments 11 12
.venv/bin/python scripts/compare_event_delays.py --experiments 12_support 12
.venv/bin/python -m pytest -q
```

검증 hash는 원본 실행 기준이다. 다른 환경이나 코드 버전의 재실행은 새 provenance를 기록하고 원본 평가 전 기록을 덮어쓰지 않는다.


[12 설정](../configs/experiment12.json) · [대조군 설정](../configs/experiment12_support.json) · [12 지표](../results/experiment12/metrics.json) · [대조군 지표](../results/experiment12_support/metrics.json) · [12 영상별 결과](../results/experiment12/per_sequence.csv) · [대조군 영상별 결과](../results/experiment12_support/per_sequence.csv) · [맥락별 경보 비교](../results/comparison11_12_support_12/context_diagnostic.json) · [보정 간섭 진단](../results/comparison11_12_support_12/calibration_interference.json) · [대조군 대비 구간 지연](../results/comparison12_support_12/events.json) · [검증](../results/experiment12/validation.json) · [사전 계획](EXPERIMENT12_PLAN.md)
