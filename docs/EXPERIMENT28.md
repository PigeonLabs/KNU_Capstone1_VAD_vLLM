# 실험 28 — 전이 CDF의 정상 관측 모집단 정렬

## 이번 실험 결과

**정상 CDF를 실제 결합에 쓰는 관측 전이로 제한했으나, 정상 holdout 오탐은 늘고 테스트 효용은 확인되지 않았다.** 세 구성 모두 정상 FP 32→40/1,920프레임, 테스트 경보·탐지 구간은 이전과 같고 Combined ranking은 소폭 낮아졌다. 이번 변경을 성능 개선으로 채택하지 않는다.

### 질문과 변경

실험 27의 연속 관측 gate를 유지하고 정상 transition reference의 모집단만 바꿨다. 이전 상태별 CDF와 지원 부족 시의 global CDF 모두 `i>0 AND relation_valid[i-1] AND relation_valid[i]`인 정상 calibration 전이만 포함한다. 이전 상태 support가 10개 미만이면 global 관측 CDF를 사용한다. global도 10개 미만이면 명시적으로 실패한다. minimum support·midrank·strict `>`는 바꾸지 않았다.

정상 FIT 전이 확률과 grammar penalty, 관측 gate, dwell, 외형 global/crop 특징과 점수, 두 외형 CDF, 실제 bank dispatch, PCA/common rank, hold/pool/age gate와 τ=56을 보존했다. 각 정상 최종 Combined 분포에서 q99를 새로 적합했다. 미래 sample·test 길이·test label을 scoring/보정에 사용하지 않았다.

R04 정상 FIT 20 / calibration 5 / 테스트 19개, seed 42, 4프레임 sampling/causal hold다. 테스트 8,154프레임(정상 3,576 / 이상 4,578), 정상 calibration 1,920프레임(482 samples)이다. 27_hold/pool/age와 각각 비교했다. VLM/검출기/CLIP 재추론은 없고 사용자 지정 로컬 Qwen 설정은 그대로다. 원본 특징 캐시로 수행한 CPU 실험이며 전체 운용 비용·지연은 미측정이다.

### 정상 reference와 보정 가능성

| reference | 기존 전체 표본 | 이번 관측 쌍 표본 | 이번 기여 영상 수 |
|---|---:|---:|---:|
| Global | 482 | 247 | 5 |
| 이전 상태 0 | 20 | 0 | 0 |
| 이전 상태 1 | 206 | 127 | 5 |
| 이전 상태 2 | 88 | 20 | 2 |
| 이전 상태 3 | 163 | 100 | 5 |

기존 global은 첫 sample을 포함하고 상태별 CDF는 제외하므로 기존 상태 합계는 477개다. 새 global/상태별 합계는 모두 247개다. 상태 0은 global fallback으로 바뀐다. 관측 쌍과 상태별 reference를 전이 확률·grammar의 직접 계산과 midrank 산식으로 독립 재구성했다.

정상 영상 holdout의 새 global reference는 174–210개/4개 영상이다. 이전 상태 2는 영상 02를 제외하면 2개/1개 영상이라 global fallback, 영상 08을 제외하면 18개/1개 영상이라 기존 count≥10 규칙상 상태 CDF를 쓴다. **표본 수와 영상 다양성은 같지 않다.** 상세 support와 영상 목록을 JSON에 기록했다.

전체 정상 및 30개 holdout 구성은 finite/[0,1]/q99<1과 최소 global support를 만족했다. 따라서 사전 기준에 따라 세 새 후보 모두 테스트했다. 전체 정상 Combined=1은 0개이며 경보 4/482 samples다. 정상 q99는 이전과 같았다: hold 0.997457627118644, pool 0.9968879668049793, age 0.9972375690607734. gate 간에는 서로 다른 q99다.

### 정상 holdout의 실패

| 제외 정상 영상 | 27_hold→28_hold | 27_pool→28_pool | 27_age→28_age |
|---|---:|---:|---:|
| 02 | 20→28 | 8→16 | 20→28 |
| 08 | 0→0 | 0→0 | 0→0 |
| 10 | 0→0 | 0→0 | 0→0 |
| 12 | 0→0 | 16→16 | 4→4 |
| 15 | 12→12 | 8→8 | 8→8 |
| 합계 / 1,920 | 32→40 | 32→40 | 32→40 |

FPR은 1.67%→2.08%다. 추가된 8프레임은 모두 영상 02의 `[152,156)`, `[376,380)`이며, 연속 관측된 `3→2` 전이다. 각 fold의 old/new q99는 같다.

| 반례의 값 (두 sample 공통) | 값 |
|---|---:|
| 나머지 4개 영상의 이전 상태 3 reference | 88개 |
| reference 최대 raw transition | 3.8590212280 |
| 제외 영상의 3→2 raw transition | 4.6699514442 |
| 기존 보정 transition | 0.9926470588 |
| 이번 보정 transition | 1.0 |

이 두 곳은 **global fallback이 아니라 지원되는 상태 3 CDF**에서 발생했다. 표본이 88개 있어도 특정 정상 전이를 포함하지 못하면 empirical CDF가 1을 출력한다. 전체 정상 calibration의 비포화가 제외 영상의 비포화를 보장하지 않는다. 이 사후 진단에서 임계값/reference를 변경하지 않았다. [정상 반례의 원인 추적](../results/experiment28/normal_alarm_attribution.json)

### R04 개발 테스트

| 구성 | Visual AUROC / AP | Combined AUROC / AP | 정상 오탐률 (FP) | 이상 recall (TP) | 탐지 / 26 |
|---|---:|---:|---:|---:|---:|
| 27_hold | 0.6899 / 0.6899 | 0.6820 / 0.6816 | 8.19% (293) | 13.65% (625) | 13 |
| 27_pool | 0.6875 / 0.6857 | 0.6797 / 0.6774 | 7.41% (265) | 11.97% (548) | 14 |
| 27_age | 0.6898 / 0.6891 | 0.6820 / 0.6808 | 8.19% (293) | 13.83% (633) | 13 |
| 28_hold | 0.6899 / 0.6899 | 0.6812 / 0.6812 | 8.19% (293) | 13.65% (625) | 13 |
| 28_pool | 0.6875 / 0.6857 | 0.6787 / 0.6768 | 7.41% (265) | 11.97% (548) | 14 |
| 28_age | 0.6898 / 0.6891 | 0.6812 / 0.6804 | 8.19% (293) | 13.83% (633) | 13 |

![짝지은 비교](../results/experiment28/observed_cdf_comparison.png)

세 짝 모두 테스트 경보 배열이 정확히 같아 추가/제거 FP/TP가 각각 0개다. 자체 q99와 기존 q99를 같은 새 점수에 적용한 결과도 같다. 탐지/미탐은 hold 13/13, pool 14/12, age 13/13이며 기존 탐지 집합과 같다. 탐지된 구간만의 지연 중앙값은 75/64.5/75프레임이고 onset 이전 활성 경보는 각각 2개다. 미탐을 제외한 조건부 지연이며 FPS 가정·point adjustment는 없다.

Combined AUROC/AP는 세 구성 모두 소폭 낮아졌다. Process AUROC/AP도 0.5679/0.5981→0.5648/0.5956으로 낮아졌고 Visual은 정확히 같다. 연속 관측 subset의 Combined AUROC/AP는 0.6358/0.6874→0.6340/0.6864다. 작은 수치 차이를 통계적 유의성으로 주장하지 않는다.

세 gate 모두 연속 관측 subset에서 Combined score가 764프레임 바뀌었다. 첫 sample·누락·재관측에서는 gate가 닫혀 있어 최종 process/Combined score가 정확히 같다. 관측 상태별/age별 지표와 이전 상태·fallback별 진단을 JSON에 기록했다.

### Support와 공정의 기여

| 새 공정 경로 | 정상 / 이상 프레임 | hold의 정상 / 이상 경보 | transition=1 프레임 |
|---|---:|---:|---:|
| 관측 쌍·상태 CDF 지원 | 1900 / 3043 | 169 / 294 | 8 |
| 관측 쌍·global fallback | 26 / 6 | 8 / 4 | 12 |
| gate 닫힘 | 1650 / 1529 | 116 / 327 | 0 |

Global fallback은 이전 상태 0의 32프레임이다. 구성 간 support subset은 같지만 경보는 외형 gate에 따라 다를 수 있다. 이전 상태별 표본 수 변경이 경보 변경을 뜻하지 않으며 이 테스트에서 짝지은 기존 경보는 모두 같다. fallback subset이 32프레임뿐이므로 성능 순위나 일반적 유용성을 추정하지 않는다.

각 구성의 자체 Combined q99에서 Visual 대비 공정 추가 경보는 정상 16/이상 4프레임으로 유지됐다. 전부 transition 기여이고 dwell 단독/공동 추가 경보는 0개다. Visual 대비 새로 탐지한 이상 구간도 0개다. 이 비교는 Visual의 임계값을 별도로 적합한 독립 baseline 실험이 아니다.

전이가 Visual·dwell보다 높은 점수를 만드는 범위는 정상 494/이상 258→510/274프레임으로 늘었으나 경보는 같았다. dwell의 점수/지원 범위(417/8,154, 5.11%)는 보존됐다.

## 결과의 의의

추론 gate와 정상 보정 모집단을 일치시키는 것만으로 성능과 안정성이 좋아지지 않는다는 반례를 확보했다. 전이 count support와 정상 전이 커버리지를 구분해야 하며, 정상 영상 holdout이 전체 calibration에서 보이지 않는 포화·오탐을 드러냈다.

이번 가치는 구성 요소의 역할과 실패 원인을 분리한 구현·검증에 있다. 성능 우위나 novelty 확보로 표현하지 않는다. 코드·결과는 비교용으로 보존하고, 점수 규칙을 더 바꾸기 전에 정상 전이의 시각적·영상별 근거를 추적한다.

## 보완할 점

- 정상 오탐이 늘고 test 효용은 없다. q99 조정으로 실패를 숨기지 않았다.
- 지원되는 상태 CDF도 특정 전이를 놓칠 수 있다. 88개 표본의 다양성·영상별 분포와 latent phase 전이의 의미를 확인해야 한다.
- 상태 2 reference는 20개/2개 영상이고 fold에서는 2개 또는 18개/1개 영상까지 줄었다. 관측이 semantic 정확도를 뜻하지 않으며 sparse transition과 오검출을 구분하지 못한다.
- FIT 전이 확률에는 전체 전이가 남아 모집단 차이가 완전히 해소된 것은 아니다. 이번 결과는 CDF만 변경한 효과다.
- R04 반복 개발·정상 calibration 5개·단일 seed, 독립 녹화 그룹/bbox·phase GT/FPS 부재를 유지한다. 다른 장면의 11개 라벨 불일치는 정렬 미확정이다. 비용·통계적 불확실성·다른 장면 일반화는 미측정이다.

## 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **정상 전이의 영상별 커버리지·관측 근거 감사**: 전이→역할/track/관계 특징의 추적 정보를 추가 | 정상 영상 02의 관측 전이 3→2 두 곳이 다른 4개 영상의 CDF 최대를 넘어 오탐 +8. 이전 상태 support 88개만으로 전이 커버리지를 보장하지 못함 | 정상 FIT/calibration 전체 전이별 표본·영상 수와 사례를 재구성. score/threshold를 바꾸지 않고 반례를 추적. 시각 검토를 semantic GT/정확도로 부르지 않음 |
| 2 | **정상 FIT 전이 확률의 관측 모집단 정렬**: 관측 쌍만으로 확률 적합하는 대조 | gate·CDF는 관측 쌍인데 FIT 전이 빈도는 전체 1,940개를 사용하며 관측 쌍은 964개 | 한 번에 FIT 확률만 변경하고 CDF/지원 부족/정상 holdout 효과를 분리. 희소 전이·가림에 의한 정당한 이상 신호 손실 검사 |
| 3 | **고정 설정의 다른 장면 적용성 검증** | 반복 R04에서 calibration만 바꿔도 정상 오탐이 증가하고 test 효용은 없음 | 장면/protocol을 먼저 고정하고 정상 적합 후 test 한 번 평가. object/phase 의미 차이, 지원 부족과 실패도 보고 |

1순위를 [실험 29 계획](EXPERIMENT29_PLAN.md)으로 구체화한다. 정상 영상의 전이 근거를 추적하는 진단 출력을 파이프라인에 추가하고, 이상 점수·임계값은 유지한다. 후속 전체 실험을 미리 확정하지 않는다.

## 검증과 재현

103개 테스트 통과. 관측/누락·영상 경계·부족 지원·잘못된 설정을 검사했다. 정상 30개 holdout의 reference/q99/예측·제외 영상 비누출, 44개 특징 hash, 테스트 114개 예측의 점수/라벨/지표/분기·구간을 재구성했다. 원래 FIT 모델·외형 점수·raw transition·관측 gate·dwell이 보존됐고 정상 reference를 독립 계산했다. 그림은 저장 JSON/CSV로 생성해 렌더링을 확인했다.

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=.cache/matplotlib
.venv/bin/python scripts/prepare_observed_transition_cdf.py
.venv/bin/python scripts/audit_observed_transition_cdf.py
.venv/bin/python scripts/evaluate_route_holdout.py --experiments 27_hold 27_pool 27_age 28_hold 28_pool 28_age --output-experiment 28 --feature-source-experiment 28
.venv/bin/python scripts/prepare_observed_transition_cdf.py --pre-test
# results/experiment28/test_eligibility.json에서 세 후보 통과 확인
for variant in hold pool age; do
  .venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset --config "configs/experiment28_${variant}.json"
done
.venv/bin/python scripts/validate_observed_transition_cdf.py
.venv/bin/python scripts/diagnose_observed_cdf_holdout.py
.venv/bin/python scripts/plot_observed_transition_cdf.py
.venv/bin/python -m pytest -q
```

사전 hash는 원본 실행 provenance다. 계획 문서는 작성 당시 상태를 보존하며 완료 상태는 본 보고서/README를 따른다. 다른 환경에서는 입력 캐시/라벨 경로와 새 실행 기록이 필요하다. 원본 영상·특징·가중치·로그는 업로드하지 않는다.

[6개 구성 CSV](../results/experiment28/comparison.csv) · [전체/paired/subset/support/분기/구간](../results/experiment28/observed_transition_cdf_diagnostic.json) · [정상 감사](../results/experiment28/normal_audit.json) · [정상 holdout](../results/experiment28/normal_holdout.json) · [테스트 자격 판정](../results/experiment28/test_eligibility.json) · [검증](../results/experiment28/validation.json) · [사전 protocol](../results/experiment28/pre_normal_protocol.json) · [테스트 전 기록](../results/experiment28/pre_test_checkpoint.json)

- 28_hold: [설정](../configs/experiment28_hold.json) · [지표](../results/experiment28_hold/metrics.json) · [영상별 결과](../results/experiment28_hold/per_sequence.csv)
- 28_pool: [설정](../configs/experiment28_pool.json) · [지표](../results/experiment28_pool/metrics.json) · [영상별 결과](../results/experiment28_pool/per_sequence.csv)
- 28_age: [설정](../configs/experiment28_age.json) · [지표](../results/experiment28_age/metrics.json) · [영상별 결과](../results/experiment28_age/per_sequence.csv)
