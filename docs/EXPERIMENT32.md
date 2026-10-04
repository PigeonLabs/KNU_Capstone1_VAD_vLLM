# 실험 32 — 객체 쌍의 연속성을 반영한 정상 체류 자료

상태: 완료. 정상 FIT/calibration 25개 영상의 episode 추출기와 출처 추적을 구현하고 독립 재구성·인과적 prefix 검증을 마쳤다. [사전 계획](EXPERIMENT32_PLAN.md)은 실행 전 동결 문서로 유지한다.

## 1. 이번 실험 결과

### 질문·범위·고정 조건

같은 객체 쌍의 phase 진입을 관측한 뒤 끝까지 볼 수 있었는가? 아니면 객체 교체·누락·영상 끝 때문에 추적만 중단된 것인가? 실험 31에서는 전이를 같은 track 쌍으로 제한했지만 체류에는 이 조건을 적용하지 않았고, 정상 완결 run의 identity 근거가 부족했다.

이번에는 **새 anomaly score·체류 분포·q99를 적합하지 않았다.** 실험 30 reset 특징/선택/phase를 사용해 정상 체류 자료를 재구성했다. 실험 31의 정상 모델·점수 및 정상 입력 총 103개 파일의 hash가 그대로다. 테스트 영상·feature·라벨·예측 파일을 열지 않았고 모델 호출도 없다. 새 AUROC/AP·FPR·recall·탐지 지연은 **미평가**이며 이전 성능을 이번 성능으로 재사용하지 않는다. 실행 비용도 미측정이다.

| 범위 | 영상 | source frames | samples | 유효 관계 samples | 미관측 samples |
|---|---:|---:|---:|---:|---:|
| FIT | 20 | 7,812 | 1,960 | 1,123 | 837 |
| calibration | 5 | 1,920 | 482 | 297 | 185 |
| 합계 | 25 | 9,732 | 2,442 | 1,420 | 1,022 |

기존 R04 sequence split·seed 42·stride 4를 유지했다. FIT와 calibration을 따로 집계하며 support를 합치지 않았다. 시간 단위는 source frame index다. FPS/timestamp·객체/semantic phase GT·독립 녹화 그룹은 여전히 없다. 전체 데이터셋의 라벨 길이 불일치 11개 시퀀스도 정렬 미확정이며 원본을 수정하지 않았다.

### 새 episode 규칙

- 같은 쌍이 연속 관측되는 phase 변경만 **관측된 진입**으로 기록한다. 이전 phase는 entry context다. 실제 action의 진입 정답이 아니다.
- 같은 쌍으로 phase가 유지되다가 같은 쌍의 다음 phase를 관측하면 **complete**다. 길이는 두 관측 경계의 index 차이이며 sample 해상도의 잠재 상태 길이다.
- 진입을 알지만 객체 쌍 교체·관계 누락·영상 종료 때문에 끝을 못 보면 **right-censored(우측 검열)**다. `마지막 실제 관측 − 진입 관측`을 보수적 길이 하한으로 저장한다. 다음 missing/교체 frame까지 채우거나 중단을 종료로 바꾸지 않는다.
- 영상 시작·재관측·객체 교체로 관측이 시작하면 **unknown-entry**다. 진입 시각/맥락과 duration은 null이며, 별도 관측 구간 길이만 기록한다. 이 길이를 완전한 체류 정답으로 쓰지 않는다.
- phase 변경 경계 sample은 새 episode에 한 번만 배정하고 missing은 `episode_id=-1`이다. sample별 age/context는 과거 관측으로만 계산한다. 영상 prefix의 마지막 episode가 임시로 검열되는 것과 미래에 완결되는 것을 구분한다.

### 전체 결과

| 파티션 | 전체 episode | complete / 영상 수 | right-censored / 영상 수 | unknown-entry / 영상 수 | 검열 하한 0 |
|---|---:|---:|---:|---:|---:|
| FIT | 257 | 7 / 3 | 22 / 13 | 228 / 20 | 8 |
| calibration | 78 | 5 / 3 | 7 / 5 | 66 / 5 | 1 |

FIT의 **88.72%**, calibration의 **84.62%**는 진입을 확인하지 못했다. 같은 쌍으로 관측한 진입을 가진 FIT 29개 중 22개가 중단되었다. 이는 독립 episode 확률 추정이 아닌 현재 자료의 기술 통계다.

![정상 체류 자료](../results/experiment32/duration_evidence.png)

### 맥락별 학습 근거

`?`는 진입 맥락 미확인이며 숫자 phase로 대체하지 않는다.

| 파티션 / 맥락 | complete / 영상 수 | 검열 / 영상 수 | 양의 검열 하한 / 0 하한 | unknown-entry |
|---|---:|---:|---:|---:|
| FIT 1→3 | 3 / 2 | 19 / 13 | 12 / 7 | 0 |
| FIT 3→1 | 4 / 3 | 3 / 3 | 2 / 1 | 0 |
| FIT ?→0 / ?→1 / ?→2 / ?→3 | 해당 없음 | 해당 없음 | 해당 없음 | 1 / 122 / 53 / 52 |
| calibration 1→3 | 1 / 1 | 6 / 5 | 5 / 1 | 0 |
| calibration 3→1 | 4 / 3 | 1 / 1 | 1 / 0 | 0 |
| calibration ?→1 / ?→2 / ?→3 | 해당 없음 | 해당 없음 | 해당 없음 | 29 / 16 / 21 |

FIT의 1→3 complete는 영상 13(1개), 14(2개), 3→1 complete는 13(1개), 14(2개), 25(1개)에 있다. 전체 complete 7개 중 4개가 영상 14에 집중된다. 두 맥락 모두 기존 **최소 10 complete runs** 조건에 미달한다. 이 기준을 낮추거나 calibration 5개를 FIT에 합치지 않았다. 검열 22개가 있다고 complete support가 29개가 되는 것은 아니다.

실험 31 감사의 같은 쌍 complete 기록과 FIT 7개/calibration 5개 모두 start/end/length까지 정확히 일치했다. 신규 extractor가 완결 run 수를 임의로 늘리거나 줄이지 않았다.

### 완결 길이와 관측 하한의 차이

| FIT 맥락 | 완결 길이 min / median / max | 검열 하한 min / median / max | 완결 max보다 큰 검열 하한 |
|---|---:|---:|---:|
| 1→3 | 12 / 12 / 16 | 0 / 4 / 72 | 5 / 19 |
| 3→1 | 4 / 4 / 12 | 0 / 12 / 16 | 1 / 3 |

단위는 source frames다. 양의 검열 하한만 보면 FIT 전체 14개는 min/median/max=4/16/72다. 0 하한 8개는 진입 sample만 보고 중단된 경우로 유지했으며 임의의 양의 길이로 바꾸지 않았다.

1→3의 하한 28·36·72·24·28프레임은 완결 표본 최대 16보다 크다. 끝을 관측한 것만 남기면 이미 확인한 긴 유지 구간의 정보를 잃을 수 있다. 그렇다고 해당 하한을 실제 종료 길이로 넣을 수는 없다. calibration의 길이 분포·모든 사례는 [기계 판독 결과](../results/experiment32/evidence_diagnostic.json)에 별도 기록했다.

### 관측이 부족해진 경로

| FIT 진입 미확인 episode의 시작 | 수 |
|---|---:|
| 관계 재관측 | 156 |
| target track 교체 | 45 |
| anchor track 교체 | 24 |
| 영상 시작 | 3 |

진입이 확인된 FIT 우측 검열 22개의 중단 원인은 관계 누락 11, target 교체 6, anchor 교체 3, 영상 끝 2다. calibration 검열 7개는 관계 누락 6/target 교체 1이다. 이는 메타데이터에서 확인한 경계 유형이며 실제 가림, 검출 누락, 역할 filter의 오제외, 물리적 객체 교체 중 무엇인지 아직 판정하지 않았다.

유효 관계 FIT 1,123 samples는 complete 구간 16, 검열 구간 91, 진입 미확인 구간 1,016으로 정확히 배정됐다. calibration은 7/22/268=297 samples다. 정상 관측 sample 합계는 보존되며 missing은 어떤 episode에도 귀속되지 않는다.

## 2. 실험 결과의 의의

체류 학습 입력을 하나의 duration 숫자로 축약하기 전에 **관측된 사건·관측 중단·진입 불확실성을 분리하는 파이프라인 구성 요소**를 구현했다. 각 episode를 선택 track·phase·source index와 sample assignment로 추적할 수 있고, 미래의 완료 여부가 과거 age/context를 바꾸지 않도록 검증했다.

지원 부족이 단순한 표본 개수 문제가 아니라 진입 미관측과 중단의 구조적 문제임을 확인했다. 특히 긴 검열 하한을 버리고 짧은 완결만 사용하는 학습의 한계를 자료에서 드러냈다. 이것이 곧 특정 생존 모형의 적합성이나 검출 개선을 증명하지는 않는다. 기존 연구 대비 novelty·의미 정확도·일반화·통계적 유의성은 주장하지 않고 넓은 학부 캡스톤 파이프라인 구현 범위를 유지한다.

## 3. 보완할 점

- 어느 맥락도 기존 complete support를 충족하지 못한다. 새 분포를 적합하거나 성능을 평가하지 않았으므로 성능 개선 여부는 미확인이다.
- 진입 미확인이 FIT 228/257개로 대부분이다. 현재 메타데이터는 중단이 있었다는 사실만 알려 주며 원인이 raw 후보 부재인지, semantic/면적 gate인지, 역할/track 혼동인지는 분해하지 않았다.
- 검열이 가림·객체 이동·공정 상태와 연관될 수 있다. 독립적 관측 중단이라고 가정할 근거가 없으며 단순 생존 추정을 적용하면 편향이 남을 수 있다.
- unknown-entry의 관측 길이를 완전한 duration으로 취급할 수 없다. zero 하한은 양의 체류 정보가 없고, 검열 하한은 종료 시각이 아니다.
- phase는 잠재 cluster, track은 추정 ID다. 같은 ID의 역할 오류가 남을 수 있다. stride 4에서 경계 사이 실제 움직임을 알 수 없고 초 단위 변환도 불가능하다.
- complete 7개가 3개 영상에만 집중한다. 정상 25영상과 반복 R04로 일반화를 주장할 수 없으며 비용·검출/localization 정확도는 미측정이다.

## 4. 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **관계 관측 손실의 원인을 분해하고 정상 역할 근거 검증**: raw 객체 후보→semantic/면적 gate→선택/track→episode 중단의 추적 정보를 추가 | FIT 228/257 episode가 진입 미확인. 이 중 재관측 시작 156개, track 교체 69개이며 진입을 아는 우측 검열 22개 중 관계 누락 11개·track 교체 9개 | 정상 25영상에서 후보가 원래 없었는지, filter로 제외됐는지, 다른 track으로 바뀌었는지 구분하고 사전 선택한 정상 사례를 검토. 캐시 후보 부재를 검출 false negative로 단정하거나 목적 표집을 의미 정확도로 계산하지 않음 |
| 2 | **검열 정보를 보존하는 체류 추정의 지원성·민감도 대조**: 완결만 쓰는 추정과 중단 전 관측 하한을 활용하는 후보 비교 | FIT 1→3은 complete 3개/검열 19개이고 그중 5개 하한이 완결 최대 16프레임보다 큼. 3→1은 complete 4개/검열 3개 | 정상 데이터만으로 영상별 민감도·꼬리 불확실성·실패를 보고. 검열의 독립성은 미확인이고 0 하한은 양의 duration으로 대체하지 않음. 기존 최소 complete support 미달을 임의 완화해 성능을 만들지 않음 |
| 3 | **다른 장면·별도 정상 자료에서 고정 파이프라인 적용성 검증** | FIT complete 7개가 영상 3개에만 있고 4개는 영상 14에 집중. 반복 R04만으로 안정적인 체류 학습을 보장할 수 없음 | 장면/분할/설정을 먼저 고정하고 calibration을 FIT에 합치지 않음. 객체·phase 의미 차이, 영상 그룹 독립성, 지원 실패까지 보고 |

1순위를 [실험 33 계획](EXPERIMENT33_PLAN.md)으로 구체화했다. 다음 변경은 낮은 관측 가용성의 원인을 먼저 분해한 뒤 선택한다. 다른 두 후보를 확정된 후속 실험으로 예약하지 않는다.

## 검증·산출물·재현

- 단위 테스트 **132개 통과**. 새 episode 테스트 13개는 complete 경계 귀속, anchor/target/both 교체, missing/reacquired, 0 하한, unknown-entry, 빈 입력, 시간/phase 유효성, prefix를 다룬다.
- 정상 25영상, sample **2,442개**, episode **335개**를 별도 구간 그룹화 알고리즘으로 재구성했다. 모든 prefix **2,467개**에서 과거 assignment/age/context와 이미 관측한 경계를 확인했다. 미래에 완결될 episode의 prefix 끝은 임시 검열로 남긴다.
- 입력/정상 모델/점수 **103개** hash 불변, 기존 complete **12개** 정확히 재현. numpy/media 데이터 접근은 정확한 정상 파일 allowlist와 test/prediction 경로 차단 hook 아래 수행했다. [추출 접근 기록](../results/experiment32/audit_access_log.json), [검증 접근 기록](../results/experiment32/validation_access_log.json).
- 첫 추출은 완료됐지만 검증 도중 guard가 `numpy/testing` 라이브러리 코드 import를 test 데이터로 오인해 중단했다. 검증 라이브러리를 guard 설치 전에 import하도록 수정했다. extractor·guard 규칙·설정·입력은 바꾸지 않았고 최초 episode/summary hash와 재실행 결과가 같다. [최초 protocol](../results/experiment32/pre_audit_attempt01.json), [수정 기록](../results/experiment32/validation_repair.json), [최종 protocol](../results/experiment32/pre_audit_protocol.json)을 보존했다.
- [설정](../configs/experiment32.json), [summary](../results/experiment32/summary.json), [모든 episode](../results/experiment32/episodes.json), [맥락별 CSV](../results/experiment32/context_evidence.csv), [검증](../results/experiment32/validation.json).
- sample assignment는 ignored `artifacts/experiment32/`에 저장했다. 원본 영상·가중치·feature cache·기존 점수 배열·로그를 업로드하지 않는다. 공개 JSON은 episode 출처와 집계 메타데이터이며 figure는 집계값만 사용한다.

실험 30 reset의 정상 특징과 실험 31의 정상 모델/점수·provenance 감사가 필요하다. hash checkpoint는 해당 실행의 증거이므로 변경된 코드/환경의 재현은 별도 실행 디렉터리에서 수행한다.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/experiment32_duration_evidence.py prepare
PYTHONPATH=src .venv/bin/python scripts/experiment32_duration_evidence.py audit
PYTHONPATH=src .venv/bin/python scripts/validate_duration_evidence.py
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/plot_duration_evidence.py
```
