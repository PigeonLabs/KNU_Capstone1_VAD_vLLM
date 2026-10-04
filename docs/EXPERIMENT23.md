# 실험 23 — 표본 수와 PCA rank를 함께 맞춘 관측 FIT 대조

## 이번 실험 결과

**rank를 같게 맞춰도 관측 FIT의 외형 ranking·낮은 오탐과 적은 구간 탐지라는 상충이 유지됐다.** 모든 구성의 Combined AUROC 변화는 절댓값 0.001 미만이었다. 관측 FIT의 경보 프레임은 이전과 완전히 같았다. rank 통제는 비교의 근거를 보강했지만 새 성능 개선으로 채택할 결과는 아니다.

### 질문·변경·고정 조건

실험 22에서 같은 수의 정상 FIT 표본을 사용해도 두 phase bank의 rank가 달랐다. 이번에는 관측 FIT(21_fit)와 random seed 0–4(22_s0–s4)의 표본 선택을 보존하고 **각 지원 phase bank의 정상 FIT rank 최솟값**을 공통 rank로 정했다. 학습 평균과 basis의 앞쪽 방향은 그대로 두고 초과 방향만 제거했다.

| bank | 관측 FIT 기존 rank | random 기존 rank 범위 | 이번 공통 rank |
|---|---:|---:|---:|
| 전체 frame(−1), phase 2 | 30 | 31–32 | **30** |
| 객체 역할 1, phase 2 | 31 | 29–32 | **29** |
| 나머지 지원 phase bank 10개 | 32 | 32 | **32** |

phase 0은 역할마다 3개 표본이라 계속 미지원이다. 지원 phase bank 12개와 pooled bank 4개의 표본 수·지원·추론 경로를 유지했다. pooled PCA는 모두 동일하므로 변경하지 않았다. 공통 rank는 정상 FIT 정보만으로 도출해 calibration 전에 고정했다. 기존 PCA의 95%/최대 rank 32 규칙을 먼저 적용한 뒤 방향 수를 제한하므로 모든 bank가 95% 분산을 설명한다고 주장하지 않는다.

새 구성은 23_obs, 23_s0–s4이며 각 기존 버전도 재현해 짝지어 비교했다. B(미관측 즉시 pooled)는 계속 꺼져 있고, 최소 지원 미달 fallback은 유지한다. 원본 phase/관측 mask·특징·bbox·track·전이/체류는 모두 같다. VLM/검출기/CLIP 재추론은 없으며 로컬 Qwen 설정도 유지했다.

범위는 R04 FIT 20 / calibration 5 / 테스트 19개다. 테스트 8,154프레임(정상 3,576 / 이상 4,578), calibration 1,920프레임(482 samples), seed 42의 기존 관계 모델과 선택 seed 0–4를 사용했다. 표본을 새로 선택하지 않고 동일 seed의 결정적 재구성 인덱스가 이전 저장값과 같은지 검사했다. 4프레임 sampling·causal hold, 역할 전체 CDF, max 결합, 정상 q99/strict `>`를 유지했다. 시간 단위는 원본 프레임이며 운용 지연/실행 비용은 새로 측정하지 않았다.

### 정상 영상 검증

정상 FIT는 고정한 채 calibration 영상 하나씩을 제외해 나머지 4개로 모든 CDF/공정 reference/q99를 적합했다. 아래는 새 공통 rank 구성의 오탐 프레임 수다.

| 제외 정상 영상 | obs | s0 | s1 | s2 | s3 | s4 |
|---|---:|---:|---:|---:|---:|---:|
| 02 | 28 | 20 | 16 | 20 | 20 | 16 |
| 08 | 0 | 4 | 4 | 4 | 4 | 4 |
| 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12 | 0 | 0 | 12 | 4 | 24 | 12 |
| 15 | 12 | 8 | 8 | 8 | 8 | 8 |
| 합계 / 1,920프레임 | 40 | 32 | 40 | 36 | 56 | 40 |

이전과 비교하면 s2의 정상 영상 08 오탐만 8→4프레임으로 줄었다. 나머지 모든 fold의 오탐 수는 같았다. 관측 FIT 40개(2.08%)는 random 32–56개(1.67–2.92%), 평균 40.8개 범위 안이다. 모든 후보를 유지했고 정상 holdout으로 seed를 선택하지 않았다.

전체 정상 보정에서는 여섯 후보 모두 finite/[0,1], 점수 1 포화 0개, 482 samples 중 경보 4개였다. q99는 s3만 0.9972375690607734, 나머지 0.997457627118644로 각 기존 버전과 같다. 각 모델의 역할 CDF/q99를 따로 적합했으며 테스트 FPR을 같게 맞춘 비교가 아니다.

### R04 개발 테스트 결과

| 공통 rank 구성 | Visual AUROC / AP | Combined AUROC / AP | 정상 오탐률 (FP) | 이상 recall (TP) | 탐지 구간 / 26 |
|---|---:|---:|---:|---:|---:|
| 23_obs | 0.7001 / 0.6971 | 0.6765 / 0.6753 | 8.64% (309) | 13.74% (629) | 13 |
| 23_s0 | 0.6912 / 0.6836 | 0.6737 / 0.6673 | 9.31% (333) | 13.48% (617) | 16 |
| 23_s1 | 0.6950 / 0.6872 | 0.6774 / 0.6712 | 9.31% (333) | 12.41% (568) | 16 |
| 23_s2 | 0.6886 / 0.6833 | 0.6715 / 0.6679 | 10.88% (389) | 16.89% (773) | 18 |
| 23_s3 | 0.6902 / 0.6832 | 0.6748 / 0.6687 | 9.20% (329) | 13.19% (604) | 16 |
| 23_s4 | 0.6904 / 0.6843 | 0.6744 / 0.6689 | 8.86% (317) | 12.93% (592) | 16 |

![공통 rank 비교](../results/experiment23/common_rank_comparison.png)

관측 FIT의 Visual AUROC 0.700126은 random 평균 0.691098/범위 0.688645–0.695039보다 높고, Visual AP·Combined AP도 다섯 random보다 높다. 정상 오탐 309개도 random 317–389개보다 적다. 반면 Combined AUROC 0.676535는 random 0.671497–0.677439 범위 안이고, 구간 탐지 13개는 모든 random의 16–18개보다 적다.

같은 표본 수와 rank에서 차이가 유지된다는 근거는 강화됐다. 하지만 관측 정보의 실제 의미 품질이나 일반적인 성능 우위를 입증한 것은 아니다. 동일 R04에 대한 다섯 seed의 평균·범위는 표본 선택 민감도이며 독립 반복/신뢰구간/통계적 유의성이 아니다.

### 동일 표본에서 rank만 바꾸었을 때

| 변경 전→후 | Δ Visual AUROC | Δ Combined AUROC | 정상 FP 변화 | 이상 TP 변화 | 구간 탐지 변화 |
|---|---:|---:|---:|---:|---:|
| 21_fit→23_obs | -0.000003 | +0.000037 | 309→309 | 629→629 | 13→13 |
| 22_s0→23_s0 | -0.000029 | +0.000033 | 333→333 | 625→617 | 17→16 |
| 22_s1→23_s1 | -0.000448 | -0.000648 | 353→333 | 620→568 | 16→16 |
| 22_s2→23_s2 | -0.000274 | -0.000693 | 389→389 | 785→773 | 18→18 |
| 22_s3→23_s3 | +0.000771 | +0.000921 | 329→329 | 588→604 | 16→16 |
| 22_s4→23_s4 | -0.000332 | -0.000549 | 317→317 | 588→592 | 17→16 |

rank 조정에 따른 Combined AUROC 변화는 −0.000693~+0.000921로 작고 방향도 일정하지 않다. 23_obs는 ranking이 조금 달라졌지만 임계값 경보는 한 프레임도 바뀌지 않았다. s0는 R04_04 `[261,379)`, s4는 R04_19 `[166,277)` 구간을 각각 잃었고 새로 얻은 구간은 없다. 나머지 paired 구성은 구간 탐지 집합이 같다.

각 구성에서 테스트의 global/crop 특징 **8,832개 관측**을 점검했다. raw residual이 실제 증가한 관측은 obs/s0/s1/s2/s3/s4 순서로 181/318/137/137/137/318개였으며 감소한 관측은 허용 오차 1e−12 기준 0개였다. rank가 같은 bank의 residual은 같고 평균·표본·pooled는 그대로였다. 그러나 CDF 재적합 이후의 점수·경보에는 단조성이 없다. 예를 들어 s1은 raw residual이 감소하지 않았어도 정상 경보 20개와 이상 경보 52개가 사라졌다.

### 동일 관측 subset과 공정 분기

미관측 subset은 정상 1,334 / 이상 1,221프레임, 관측 subset은 정상 2,242 / 이상 3,357프레임으로 고정했다. AP와 전체 지표는 JSON에 모두 공개했다.

| 구성 | 미관측 Visual / Combined AUROC | 관측 Visual / Combined AUROC | 미관측 정상 / 이상 경보 | 관측 정상 / 이상 경보 |
|---|---:|---:|---:|---:|
| 23_obs | 0.8077 / 0.8077 | 0.6494 / 0.6087 | 84 / 283 | 225 / 346 |
| 23_s0 | 0.7969 / 0.7969 | 0.6376 / 0.6083 | 88 / 263 | 245 / 354 |
| 23_s1 | 0.7955 / 0.7955 | 0.6451 / 0.6153 | 92 / 234 | 241 / 334 |
| 23_s2 | 0.7949 / 0.7949 | 0.6353 / 0.6067 | 108 / 311 | 281 / 462 |
| 23_s3 | 0.8007 / 0.8007 | 0.6348 / 0.6087 | 92 / 266 | 237 / 338 |
| 23_s4 | 0.7929 / 0.7928 | 0.6388 / 0.6120 | 100 / 238 | 217 / 354 |

관측 FIT의 Visual AUROC 우위는 관측·미관측 subset 양쪽에서 유지된다. 그러나 관측 subset의 Combined AUROC 0.6087은 random 범위 안이다. 모든 구성에서 process 점수는 정확히 같고 AUROC/AP는 0.5688/0.5941, 체류 유효 비율은 417/8,154(5.11%)다. 외형 ranking의 장점이 결합 점수에 그대로 반영되지는 않는다.

실제 bank 경로도 모든 구성에서 같다. 테스트 global 2,047 samples는 관측 phase 1,395 / 미관측 phase 471 / 관측 지원 부족 pooled 11 / 미관측 지원 부족 pooled 170개다. **미관측 641개 중 471개는 phase bank를 유지하며, 이 규칙은 누락 지속 길이를 구분하지 않는다.** 이 구조가 다음 개선 후보의 근거이며, 이번에 누락 길이별 효과를 검증한 것은 아니다.

### 구간 탐지와 지연

| 구성 | 탐지 / 미탐 | 탐지된 구간만의 지연 중앙값 (프레임) | onset 이전 경보 활성 탐지 |
|---|---:|---:|---:|
| 23_obs | 13 / 13 | 75 | 2 |
| 23_s0 | 16 / 10 | 70 | 2 |
| 23_s1 | 16 / 10 | 70 | 2 |
| 23_s2 | 18 / 8 | 41.5 | 2 |
| 23_s3 | 16 / 10 | 70 | 2 |
| 23_s4 | 16 / 10 | 51.5 | 2 |

새 random 다섯 구성은 23_obs가 탐지한 13개 구간을 모두 유지하고 각각 3/3/5/3/3개를 추가했다. 공통 추가 구간은 R04_11 `[170,240)`, R04_15 `[54,376)`이다. 구간 수가 많아도 정상 오탐이 늘고 각 이상 구간 내부의 경보 비율은 다르므로 종합적인 승자로 선택하지 않는다.

지연은 탐지된 구간만의 조건부 중앙값이고 미탐은 별도 개수로 남긴다. 탐지 집합이 다르므로 이 값을 전체 탐지 속도로 해석하지 않는다. onset 이전 경보 활성 여부를 표시했으며 point adjustment·FPS 가정은 없다. 모든 구간과 paired gain/loss를 JSON에 공개했다.

## 결과의 의의

표본 수뿐 아니라 rank·bank 지원·추론 경로를 맞추어도 관측 FIT와 random FIT의 차이가 남았다. 실험 22의 차이를 PCA 용량 차이만으로 설명하기 어렵다는 제한된 근거다. 정상 학습 표본 구성과 외형 모델의 기여를 재현 가능한 방식으로 비교하는 파이프라인 검증을 보강했다.

동시에 rank 조정이 오탐/구간 미탐의 상충을 해결하지 못함을 확인했다. 다음에는 실제 추론 경로의 불확실성, 특히 관계 관측이 끊긴 뒤 이전 phase를 유지하는 규칙을 개선한다. 성능 미세 변화 자체를 novelty로 주장하거나 논문 주제를 확정하지 않는다.

## 보완할 점

- 관측 FIT는 더 낮은 정상 오탐과 더 적은 탐지 구간을 함께 보인다. 정상 holdout의 우위도 일관되지 않는다.
- 같은 count/rank라도 선택된 특징의 영상별·시간적 구성과 의미적 정확성은 통제되지 않았다. 이 실험만으로 관계 관측 mask의 의미 품질이 입증되지는 않는다.
- PCA rank는 두 bank에서 1–3개 방향만 줄어든 대조다. 작은 효과를 rank가 항상 중요하지 않다는 일반적 결론으로 확장하지 않는다.
- 역할 전체 CDF와 공정 max 결합의 간접 효과가 남는다. raw residual 증가가 더 많은 경보를 의미하지 않는다.
- 반복 R04 개발, 5개 정상 calibration, bbox/semantic phase GT·독립 녹화 그룹·FPS 부재를 유지한다. 다른 장면의 라벨 불일치 11개는 정렬 미확정으로 남고 원본을 임의 수정하지 않는다.

## 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **누락 지속 길이에 따른 인과적 fallback**: 정상 FIT 누락 길이로 이전 phase 유지 한도를 정함 | count/rank 통제 후에도 관측 FIT는 13/26 구간 탐지, 미관측 global 471 samples에서 phase bank 유지 | 무기한 유지·즉시 pooled·나이 기반 전환을 같은 bank로 비교. 정상 FIT만으로 한도 고정, 길이별 FP/recall·경계 지연·CDF 간접 효과 검증. 미래 관측/테스트 라벨로 gate 결정 금지 |
| 2 | **관측 근거를 반영한 공정 점수 신뢰도**: 불확실한 상태에서 전이·체류 분기의 기여를 구분 | 모든 구성에서 동일 process AUROC 0.5688, Combined ranking이 Visual보다 낮음 | 정상 관측/지원 정보로만 규칙을 만들고 공정의 독자 TP·FP와 정상 보정을 확인. 테스트 지표로 가중치를 선택하거나 체류 가용성을 정확도로 취급하지 않음 |
| 3 | **정상 객체 역할·관계 관측 감사 확대**: 실제 부품과 혼동 객체의 추가 사례 검증 | count/rank 통제로도 잘못된 역할·관계 mask 자체는 교정되지 않음 | 개발 사례와 별도 정상 사례/보조 주석을 구분하고 오류·누락/비용 보고. mask를 semantic GT로, 반복 R04를 독립 검증으로 간주하지 않음 |

다음은 1순위만 구체화한 [실험 24 계획](EXPERIMENT24_PLAN.md)이다. 정상 FIT의 완결 누락 구간 90분위수를 유지 한도로 고정하고 무기한 유지/즉시 pooled/나이 기반 전환을 비교한다. 후속 전체 실험이나 여러 threshold sweep을 미리 기획하지 않는다.

## 검증과 재현

**82개 테스트 통과.** 공통 rank map의 완전성·범위, pooled 보호, basis prefix/평균/표본 보존, residual 비감소, 동일 rank의 무변경을 검사했다. 44개 특징 파일 byte hash, 정상 FIT 표본·PCA의 독립 재구성, 정상 FIT 최솟값으로의 rank 도출을 확인했다. 정상 holdout 60개 구성(12버전×5 fold)의 reference/q99/예측과 전체 테스트 228개 예측(12×19)의 객체/프레임 점수·원본 라벨·지표를 재현했다. 기존 여섯 버전의 결과도 저장값과 정확히 일치했다. 그림은 JSON/CSV로 생성하고 렌더링을 확인했다.

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=.cache/matplotlib
.venv/bin/python scripts/prepare_common_rank.py
.venv/bin/python scripts/audit_common_rank.py
.venv/bin/python scripts/evaluate_route_holdout.py --experiments 21_fit 22_s0 22_s1 22_s2 22_s3 22_s4 23_obs 23_s0 23_s1 23_s2 23_s3 23_s4 --output-experiment 23 --feature-source-experiment 23
.venv/bin/python scripts/prepare_common_rank.py --pre-test
for variant in obs s0 s1 s2 s3 s4; do
  .venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset --config "configs/experiment23_${variant}.json"
done
.venv/bin/python scripts/validate_common_rank.py
.venv/bin/python scripts/plot_common_rank.py
.venv/bin/python -m pytest -q
```

사전 hash는 원본 실행 provenance다. [실험 23 계획](EXPERIMENT23_PLAN.md)의 작성 당시 상태는 동결해 보존하고 현재 완료 상태는 이 보고서/README를 따른다. 과거 hash는 해당 commit 기준이다. 다른 환경에서는 입력 캐시/원본 라벨 경로와 새 실행 기록을 준비해야 한다. 원본 영상·특징·가중치·로그는 업로드하지 않는다.

[12개 버전 비교 CSV](../results/experiment23/comparison.csv) · [전체/paired/subset/구간 진단](../results/experiment23/common_rank_diagnostic.json) · [정상 rank/표본 감사](../results/experiment23/normal_audit.json) · [정상 holdout](../results/experiment23/normal_holdout.json) · [정상 실행 전 고정](../results/experiment23/pre_normal_protocol.json) · [테스트 전 기록](../results/experiment23/pre_test_checkpoint.json) · [검증](../results/experiment23/validation.json)

- 23_obs: [설정](../configs/experiment23_obs.json) · [지표](../results/experiment23_obs/metrics.json) · [영상별 결과](../results/experiment23_obs/per_sequence.csv)
- 23_s0: [설정](../configs/experiment23_s0.json) · [지표](../results/experiment23_s0/metrics.json) · [영상별 결과](../results/experiment23_s0/per_sequence.csv)
- 23_s1: [설정](../configs/experiment23_s1.json) · [지표](../results/experiment23_s1/metrics.json) · [영상별 결과](../results/experiment23_s1/per_sequence.csv)
- 23_s2: [설정](../configs/experiment23_s2.json) · [지표](../results/experiment23_s2/metrics.json) · [영상별 결과](../results/experiment23_s2/per_sequence.csv)
- 23_s3: [설정](../configs/experiment23_s3.json) · [지표](../results/experiment23_s3/metrics.json) · [영상별 결과](../results/experiment23_s3/per_sequence.csv)
- 23_s4: [설정](../configs/experiment23_s4.json) · [지표](../results/experiment23_s4/metrics.json) · [영상별 결과](../results/experiment23_s4/per_sequence.csv)
