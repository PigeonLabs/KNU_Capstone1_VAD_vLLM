# 산업 공정 영상 이상탐지 파이프라인

정상 영상으로 공정 패턴을 학습하고 객체별 외형·공정 흐름의 이상을 찾는 학부 캡스톤 연구입니다. 특정 논문 주제를 미리 고정하지 않고, 기본 파이프라인의 관측 결과를 바탕으로 다음 실험을 설계합니다.

## 진행 현황

| 단계 | 범위 | 상태 | 결과 |
|---|---|---|---|
| 데이터 정리 (Stage 00) | IPAD 16개 장면 전체 | 안전한 로더·평가 마스크 완료, 정확한 재라벨링은 미해결 | 아래 표 및 [기계 판독 결과](results/stage00/summary.json) |
| 실험 01 | R01 전체 학습/테스트를 사용하는 기본 파이프라인 pilot | 완료: R01 15개 테스트 영상 평가 | [결과](results/experiment01/metrics.json) · [설정](configs/experiment01.json) |
| 실험 02 | 정상 trajectory 기반 phase 추정 | 완료: phase만 교체한 비교 | [결과와 한계](docs/EXPERIMENT02.md) |
| 실험 03 | 정상 이동 영역 기반 제품 재검출 | 완료: 관측 개선, 최종 결합 성능 저하 | [결과·의의·추천 3개](docs/EXPERIMENT03.md) |
| 실험 04 | 정상 phase별 진행량 기반 공정 점수 | 완료: AUROC/AP·recall 상승, 오탐 증가 | [결과·의의·추천 3개](docs/EXPERIMENT04.md) |
| 실험 05 | 여러 연속 관측의 진행량 | 완료: 오탐 감소, 관측 가용성 감소 | [결과·의의·추천 3개](docs/EXPERIMENT05.md) |
| 실험 06 | 진행량 phase 조건화 제거 비교 | 완료: 단순화 가능성, 오탐 증가 | [결과·의의·추천 3개](docs/EXPERIMENT06.md) |
| 실험 07 | 실제 장면 R03 적용성 | 완료: 진행량 추가로 ranking 저하, 상태 관측 실패 확인 | [결과·의의·추천 3개](docs/EXPERIMENT07.md) |
| 실험 08 | R03 객체 관계 기반 잠재 상태 | 완료: 외형 ranking 상승, 결합 ranking 하락 | [결과·의의·추천 3개](docs/EXPERIMENT08.md) |
| 실험 09 | 이전 상태별 공정 점수 보정 | 완료: 결합 ranking 상승, 프레임 recall 하락 | [결과·의의·추천 3개](docs/EXPERIMENT09.md) |
| 실험 10 | max 결합 및 Visual 단독 비교 | 완료: 평균 결합 억제 해소, 추가 공정 기여 제한적 | [결과·의의·추천 3개](docs/EXPERIMENT10.md) |
| 실험 11 | 관측된 진입 이후 체류 시간 신호 | 완료: 결합 ranking 하락, 추가 오탐 증가 | [결과·의의·추천 3개](docs/EXPERIMENT11.md) |
| 다음 실험 12 | 진입 맥락별 체류 시간 및 동일 support 대조 | 계획 완료, 미실행 | [계획](docs/EXPERIMENT12_PLAN.md) |

각 실험을 마치면 **이번 결과 → 결과의 의의 → 보완할 점 → 다음 Recommended improvements(추천순 3개)**를 보고합니다. 각 추천에는 관측 근거·변경 내용·검증 기준을 포함하고, 다음 결과에 따라 우선순위를 갱신합니다. [보고 규칙](docs/EXPERIMENT_REPORTING.md)

## Stage 00 — 데이터 정합성과 시간축

| 항목 | 실측 |
|---|---:|
| 장면 | 합성 12 + 실제 4 |
| 정상 학습 시퀀스 / 프레임 | 776 / 430,867 |
| 테스트 시퀀스 / 프레임 | 292 / 167,112 |
| 길이가 일치하는 테스트 시퀀스 | 281 |
| 주 평가에 포함할 테스트 프레임 | 159,818 (95.64%) |
| 정렬 미확정 시퀀스 / 프레임 | 11 / 7,294 |
| 확인 가능한 시간 단위 | 원본 프레임 인덱스 |
| FPS·실제 timestamp·semantic phase GT | 확인 불가, null로 기록 |

**원본 ZIP에서도 라벨 길이 불일치가 동일하게 발견됩니다. 정확한 정렬을 복구했다고 주장하지 않습니다.** 원본 프레임·라벨은 수정하지 않습니다. 길이가 다른 시퀀스의 파생 평가 라벨은 모두 `-1`(unknown)로 저장하며 주 지표에서 제외합니다. 이 규칙은 추론·특징·phase·점수 계산에는 영향을 주지 않습니다. 전체 공식 benchmark 성능과 직접 동등한 protocol은 아닙니다.

| 시퀀스 | 프레임 | 원본 라벨 | 처리 |
|---|---:|---:|---|
| R02/12 | 806 | 805 | 정렬 미확정 |
| R02/13 | 609 | 608 | 정렬 미확정 |
| R02/14 | 497 | 498 | 정렬 미확정 |
| S05/09–15 (각각) | 626 | 676 | 정렬 미확정 |
| S12/13 | 1,000 | 1,001 | 정렬 미확정 |

1. 모든 시퀀스의 파일명을 정수 프레임 번호로 정렬하고 0부터 연속인지 검사합니다.
2. 테스트 라벨 292개는 원본 ZIP의 내용과 바이트 단위로 비교했습니다. 모든 시퀀스의 ZIP/로컬 프레임 수도 일치합니다.
3. 이미지 내용 비교·디코딩 검사는 시퀀스별 처음/중간/끝에 한정합니다. 전체 JPG 디코딩·콘텐츠 해시 검사를 완료한 것은 아닙니다.
4. FPS를 25/30으로 가정하지 않습니다. 속도·duration·탐지 지연은 프레임 단위로만 다룹니다.
5. 공식 로더의 `frame_name * 200 // sequence_length`는 상대 위치 proxy입니다. 실제 주기 경계 또는 semantic phase 정답으로 사용하지 않습니다.
6. score를 전체 프레임으로 확장할 때 이전 관측 점수를 유지합니다. 미래 샘플을 이용한 선형 보간을 하지 않습니다.
7. 정상 영상은 장면별로 약 80% fit / 20% calibration에 고정 분할했습니다. 시퀀스 단위 분할이며, 테스트 라벨로 분할·임계값을 선택하지 않습니다. 근접/파생 영상의 원본 그룹 정보는 없으므로 그룹 독립성은 아직 입증하지 못했습니다.

[전체 manifest](results/stage00/manifest.json) · [시퀀스 CSV](results/stage00/sequences.csv) · [불일치 근거](results/stage00/mismatches.json) · [정상 분할](results/stage00/splits.json)

### 재현

```bash
uv venv .venv --python python3.12
uv pip install --python .venv/bin/python -r requirements.lock.txt
export PYTHONPATH=src
.venv/bin/python scripts/audit_data.py \
  --data-root /path/to/IPAD_dataset \
  --archive /path/to/IPAD_dataset-001.zip
.venv/bin/python -m pytest -q
```

파생 라벨은 `artifacts/labels_strict/`에 생성됩니다. 원본 데이터·가중치·특징 캐시·개인 계획서·서버 로그는 GitHub에 업로드하지 않습니다.

## 실험 01의 구현 범위

정상 FIT 영상 → 로컬 Qwen 객체/phase 후보 → GroundingDINO → IoU tracking → frozen CLIP의 전체 프레임/crop 특징 → 관측 phase별 PCA → visual/process score → frame 및 객체 후보 점수.

첫 실험은 R01의 모든 시퀀스에서 실행합니다. 4프레임 간격 sampling, 정상 calibration, 현재 관측 기반 phase 추정을 사용합니다. 다른 장면의 결과나 전체 IPAD 재현으로 확대 해석하지 않습니다. 제안 파이프라인에는 원래 완전한 구현 명세가 없으므로 이 설정은 **우리의 명시적 baseline 구현**이며, 기존 논문 수치 재현은 아닙니다.

추후 개선은 실험 01 결과를 확인한 뒤 Recommended improvements로 기록합니다. phase 추정·누락 객체·좌표/관계 특징 등에 대한 개선을 미리 성능 기여로 주장하지 않습니다.

## 실험 01 결과

정상 fit 27개 / calibration 7개, 테스트 15개 영상의 **3,685프레임**을 평가했습니다. 이상 프레임은 1,254개입니다. seed 42 한 번의 R01 pilot이며 전체 IPAD 성능을 의미하지 않습니다.

| 점수 | Frame AUROC | Average precision |
|---|---:|---:|
| Visual | 0.5727 | 0.3766 |
| Process | 0.4942 | 0.3377 |
| Combined (0.5/0.5) | 0.5371 | 0.3636 |

정상 calibration q99 임계값은 0.9810입니다. 이 임계값에서 테스트 정상 프레임 오탐률은 **6.95%**, 이상 프레임 recall은 **5.98%**입니다. AUROC/AP만 보고 실사용 가능한 성능으로 해석하지 않습니다.

![실험 01 점수와 phase 분포](results/experiment01/summary.png)

![실험 01 고정 사례의 시간 점수](results/experiment01/timelines.png)

[상세 방법](docs/EXPERIMENT01.md) · [전체 지표](results/experiment01/metrics.json) · [영상별 지표](results/experiment01/per_sequence.csv) · [검출/추출 진단](results/experiment01/extraction.json)

### 실험 01의 의의

로컬 VLM부터 객체·공정 이상 점수까지 연결하는 실행 가능한 baseline을 확보했습니다. Process를 더했을 때 Visual 단독보다 AUROC가 낮아지고 중앙 phase가 사라진 결과는 phase 관측 방식의 검증이 우선임을 보여줍니다. 파이프라인 구현의 근거이며 novelty나 일반화 성능의 입증은 아닙니다.

### 실험 01의 보완할 점

- **Phase collapse:** FIT 샘플의 phase 분포가 `[1363, 0, 193]`입니다. 왼쪽/중앙/오른쪽이라는 언어적 설명만으로 CLIP 전체 프레임 특징이 중앙 상태를 분리하지 못했습니다. Calibration/Test에서도 중앙은 각각 1샘플뿐입니다. Process AUROC 0.4942와 함께 볼 때 현 phase proxy의 유효성이 부족합니다.
- **객체 오검출:** 정상 영상 01의 bbox 직접 점검에서 하단 고정 빨간 부품도 product 역할로 검출되었습니다. `role_frame_coverage=1.0`은 해당 역할의 박스가 있다는 뜻이며, 실제 제품 recall 100%가 아닙니다.
- **외형 정보의 한계:** crop 정규화로 위치 정보를 잃으며, motion/관계는 아직 scoring에 사용하지 않습니다.

### 실험 01 이후 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보 | 관측 근거와 검증 방향 |
|---|---|---|
| 1 | 정상 trajectory 기반 phase 추정 | 중앙 FIT 표본 0개. phase만 교체하고 관측률·점수·오탐/recall 비교 |
| 2 | 제품 grounding 검증·개선 | 고정 부품 오검출. bbox subset에서 precision/recall·누락 길이 확인 |
| 3 | Process 신뢰도에 따른 결합·보정 | Combined가 Visual보다 낮음. 정상 데이터로 결정한 결합 규칙 비교 |

[후보별 변경 내용·검증 기준](docs/EXPERIMENT01.md)에 따라 1순위를 실험 02로 선택했습니다.

**실험 02의 단일 변경:** detector, crop 특징, encoder, split, sampling, PCA/scoring 규칙은 그대로 유지하고 phase 추정만 바꿉니다. 정상 FIT에서 실제로 이동한 product-role track으로 이동 경로와 3개 공간 상태를 적합하고, test-time에는 현재 bbox와 과거 상태만 사용합니다. 누락 시 이전 phase를 유지하며 그 비율을 공개합니다. 위치 정답·test label·미래 프레임은 사용하지 않습니다. 이는 R01의 공간적 진행 proxy이며 일반 공정 grammar 학습으로 주장하지 않습니다. 당시 그 이후 실험은 미리 설계하지 않았습니다.

## 실험 02 결과와 비교

실험 01의 검출·encoder 특징을 재사용하고 **phase 추정만** 정상 trajectory로 교체했습니다. `phases` 이외의 입력 배열이 모두 동일한지 검사했습니다. 상세 근거는 [실험 02 문서](docs/EXPERIMENT02.md)에 있습니다.

| R01 평가 | 실험 01 | 실험 02 |
|---|---:|---:|
| Combined AUROC | 0.5371 | 0.5959 |
| Combined average precision | 0.3636 | 0.4031 |
| Process AUROC | 0.4942 | 0.5471 |
| 정상 q99에서 테스트 정상 프레임 오탐률 | 6.95% | 14.85% |
| 정상 q99에서 테스트 이상 프레임 recall | 5.98% | 23.37% |
| FIT phase 0/1/2 샘플 수 | 1363 / 0 / 193 | 998 / 444 / 114 |

![실험 01–02 비교](results/comparison01_02/comparison.png)

### 실험 02의 의의

동일한 검출·encoder 특징에서 phase 추정만 교체한 비교로 Combined AUROC +0.0588, AP +0.0395를 관측했습니다. phase 구성 방식이 후속 정상 모델에 영향을 준다는 근거입니다. semantic phase 정확도나 novelty를 입증한 결과는 아닙니다.

### 실험 02의 보완할 점

**Ranking은 개선됐지만 오탐도 증가했습니다.** 각 모델의 정상 calibration에서 결정한 q99를 그대로 적용한 결과입니다. 동일 test FPR 비교나 통계적 유의성 주장이 아닙니다.

또한 공간 phase를 직접 관측할 수 있었던 bbox는 전체 샘플의 **383/2893 = 13.24%**뿐입니다. 나머지는 이전 phase를 유지하거나 첫 관측 전 초기 phase 0을 사용했습니다. 상태 점유율이 달라졌다는 것만으로 semantic phase 정확도를 주장하지 않습니다. 현재 병목은 검출 품질·관측 누락입니다.

### 실험 02 이후 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보 | 관측 근거와 검증 방향 |
|---|---|---|
| 1 | **제품 grounding 품질 개선** | 직접 phase 관측 13.24%. 색상 의존·고정 배경 혼동을 점검하고 bbox subset의 precision/recall·누락 길이 검증 |
| 2 | **누락 관측을 반영한 phase 불확실성** | 오래된 phase 유지 가능. 관측 경과 프레임 수·unknown·재관측 처리를 비교하고 오탐/recall 확인 |
| 3 | **신뢰도 기반 결합·정상 calibration 개선** | 정상 오탐률 14.85%. 정상 데이터로 결합/임계값 규칙을 결정하고 경보 품질 비교 |

[후보별 변경 내용·검증 기준과 주의점](docs/EXPERIMENT02.md)을 기록했습니다. 다음 실험의 최우선 후보는 제품 grounding 개선이며, 세 후보를 확정된 후속 실험으로 취급하지 않습니다. 이후 1순위의 정상 이동 영역을 이용한 제품 재검출을 실험 03에서 실행했습니다. R01은 개발 장면이며 최종 검증 결과로 간주하지 않습니다.

[실험 02 지표](results/experiment02/metrics.json) · [phase 관측 진단](results/experiment02/phase_grounding.json) · [비교 CSV](results/comparison01_02/metrics.csv)

## 실험 03 결과 — 정상 이동 영역 기반 제품 재검출

정상 FIT의 이동 track으로 제품 검출 영역을 정했습니다. detector/prompt는 유지하고, 다른 역할 및 전체 프레임 특징과 phase 공간 지도는 고정했습니다. R01·seed 42, 테스트 15개 영상/3,685프레임으로 실험 02와 비교했습니다.

| 지표 | 실험 02 | 실험 03 |
|---|---:|---:|
| Visual AUROC | 0.5787 | 0.6514 |
| Visual AP | 0.3798 | 0.4260 |
| Process AUROC | 0.5471 | 0.5388 |
| Process AP | 0.3749 | 0.3603 |
| Combined AUROC | 0.5959 | 0.5814 |
| Combined AP | 0.4031 | 0.4018 |
| 정상 q99 기준 테스트 정상 오탐률 | 14.85% | 2.96% |
| 정상 q99 기준 테스트 이상 recall | 23.37% | 3.83% |
| 직접 phase 관측률 (전체 sampled 시점) | 13.24% | 94.75% |

![실험 02–03 비교](results/comparison02_03/comparison.png)

**의의:** 정상 trajectory를 검출 입력에 연결해 공간 phase 관측을 늘리고 Visual AUROC를 높일 가능성을 확인했습니다. 다만 Combined AUROC는 하락했습니다. 관측 품질 개선을 최종 탐지 개선이나 novelty 입증으로 확대 해석하지 않습니다.

**보완할 점:** 오탐 감소와 함께 recall도 23.37%→3.83%로 줄었습니다. 전이 빈도 점수에는 phase 내 실제 진행·정지 정보가 없고, 제품 없는 정상 구간의 배경 오검출도 남았습니다. 관측률은 제품 recall이 아니며 독립 bbox GT와 다른 장면 검증이 필요합니다. ROI의 경로 밖 이상 누락 가능성도 있습니다.

### 실험 03 이후 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보 | 검증 방향 |
|---|---|---|
| 1 | **정상 궤적의 진행량 기반 공정 점수** | 동일 track의 프레임당 이동량으로 실제 진행·정지 신호 추가. 정상 FIT 진단 후 규칙 고정 |
| 2 | **관측 상태와 실제 부재 구분** | 빈 벨트·검출 누락·재관측을 구분하고 정상 오탐과 제품 누락 이상을 함께 점검 |
| 3 | **공정 기여를 검증하는 결합·보정** | Visual 단독/고정 결합/정상 신뢰도 결합 비교. test 지표로 가중치 탐색 금지 |

[상세 결과·의의·한계·후보별 검증 기준](docs/EXPERIMENT03.md) · [실험 04 계획](docs/EXPERIMENT04_PLAN.md) · [지표](results/experiment03/metrics.json) · [관측 진단](results/experiment03/observations.json)

## 실험 04 결과 — 정상 phase별 진행량 추가

동일 track의 인접 관측에서 원본 프레임당 진행량을 계산하고, phase별 정상 median/MAD 편차를 기존 공정 점수에 추가했습니다. 미관측·track 변경 시 진행량은 사용하지 않습니다. 실험 03의 특징·phase·Visual 및 기존 전이 점수가 보존됐는지 검사했습니다.

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

![실험 03–04 비교](results/comparison03_04/comparison.png)

**의의:** 외형과 phase를 고정한 상태에서 시간적 진행량이 추가 이상 신호를 제공할 가능성을 확인했습니다. Combined AUROC +0.1032, AP +0.2279입니다. 이는 R01 개발 결과이며 phase 조건화의 독립 기여나 알고리즘 novelty·일반화 입증은 아닙니다.

**보완할 점:** 정상 오탐률도 2.96%→9.95%로 늘었습니다. 오탐 구간 41개 중 30개가 4프레임 이하였지만, 실제 이상과 겹치는 경보도 짧은 경우가 있어 단순 제거는 위험합니다. 진행량 유효 관측률은 테스트 sampled 시점 기준 93.85%이며 실제 부재와 검출 실패 문제는 남아 있습니다. 각 모델의 정상 q99에서 비교했으므로 동일 테스트 오탐률 비교가 아닙니다.

### 실험 04 이후 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보 | 검증 방향 |
|---|---|---|
| 1 | **여러 연속 관측으로 진행량 추정** | 단일 간격 변동을 줄이는 인과적 변위 계산. 관측률·오탐/recall·지연·짧은 이상 손실 함께 확인 |
| 2 | **phase 조건의 필요성 분리 검증** | phase별/pooled 진행량 모델만 교체해 추가 복잡성의 기여 확인 |
| 3 | **미사용 장면 적용성과 검증 분리** | 테스트 성능과 무관하게 다른 장면을 선택하고 정상 데이터로만 파이프라인 적합 |

[상세 보고서](docs/EXPERIMENT04.md) · [지표](results/experiment04/metrics.json) · [오류 진단](results/experiment04/error_diagnosis.json) · [실험 05 계획](docs/EXPERIMENT05_PLAN.md)

## 실험 05 결과 — 여러 연속 관측의 진행량

진행량 계산을 4프레임 간격에서 12프레임 변위로 변경했습니다. 연속 4개 관측의 track ID가 같을 때만 사용하며, 나머지 특징·phase·점수 규칙은 유지했습니다.

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

![실험 04–05 비교](results/comparison04_05/comparison.png)

**의의:** R01에서 정상 오탐 프레임이 242→144로 줄고 AUROC/AP가 소폭 상승했습니다. 8개 GT 이상 구간 모두 경보가 있었으며 첫 경보 지연 중앙값은 두 실험 모두 38.5프레임입니다. 각자의 정상 q99 기준으로 비교했으며 동일 테스트 오탐률 비교가 아닙니다.

**보완할 점:** 테스트 진행량 가용성은 sampled 기준 93.85%→90.40%로 줄었습니다. 구간 안에 경보가 있다는 사실은 이상 프레임 전체를 탐지했다는 뜻이 아닙니다. 12프레임 이하 GT 이상 구간은 0개여서 짧은 이상 보존은 검증하지 못했습니다. 단일 장면 개발 결과이며 novelty·일반화 입증은 아직입니다.

### 실험 05 이후 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보 | 검증 방향 |
|---|---|---|
| 1 | **진행량 phase 조건의 필요성 분리** | phase별/pooled 진행량 모델만 교체해 구성 요소의 기여와 복잡성 확인 |
| 2 | **미사용 장면·짧은 이상 검증** | 다른 공정과 이상 길이에서 정상 데이터만으로 적합하고 적용 한계 평가 |
| 3 | **초기·미관측 구간의 이상 신호 보완** | 긴 첫 경보 지연과 관측 부재를 진단하고 정상 빈 벨트/실제 누락 구분 |

[상세 보고서](docs/EXPERIMENT05.md) · [지표](results/experiment05/metrics.json) · [이벤트별 지연](results/comparison04_05/events.json) · [실험 06 계획](docs/EXPERIMENT06_PLAN.md)

## 실험 06 결과 — 진행량 phase 조건화 제거

동일한 12프레임 진행량을 phase별 모델 대신 하나의 pooled 정상 모델로 적합했습니다. 외형 PCA의 phase 조건화와 전이는 유지했습니다. 모든 입력 특징·관측 마스크·기존 branch가 보존됐는지 확인했습니다.

| 지표 | 실험 05: phase 조건 | 실험 06: pooled |
|---|---:|---:|
| Visual AUROC | 0.6514 | 0.6514 |
| Visual AP | 0.4260 | 0.4260 |
| Process AUROC | 0.6889 | 0.6929 |
| Process AP | 0.6502 | 0.6502 |
| Combined AUROC | 0.6992 | 0.7049 |
| Combined AP | 0.6584 | 0.6612 |
| 정상 q99 기준 테스트 정상 오탐률 | 5.92% | 7.82% |
| 정상 q99 기준 테스트 이상 recall | 42.26% | 45.45% |
| 첫 경보 지연 중앙값 (원본 프레임) | 38.5 | 12.0 |
| 경보가 발생한 GT 이상 구간 | 8 / 8 | 8 / 8 |
| 진행량 모델 수 (fallback 포함) | 4 | 1 |

![실험 05–06 비교](results/comparison05_06/comparison.png)

**의의:** 진행량 모델을 4→1개로 줄여도 이번 R01의 ranking은 낮아지지 않았습니다. 진행량의 phase 조건이 필수라는 주장은 현재 결과가 지지하지 않습니다. 외형 PCA의 phase 조건까지 불필요하다는 뜻은 아닙니다.

**보완할 점:** 오탐률이 5.92%→7.82%로 증가했습니다. phase 0의 이상 탐지 프레임이 0→88개로 늘면서 정상 오탐도 0→68개로 늘었습니다. 각 모델의 정상 q99가 달라졌으므로 지연 개선을 동일 오탐률에서의 우월성으로 해석하지 않습니다. AUROC/AP 차이도 작고 단일 장면 개발 결과에 한정됩니다.

### 실험 06 이후 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보 | 검증 방향 |
|---|---|---|
| 1 | **미사용 실제 장면 R03 적용성** | 장면별 실행 구조를 분리하고 정상 discovery·관측 가정을 확인한 뒤 같은 R03 특징에서 진행량 추가 전후 비교 |
| 2 | **정상 조건별 보정·오탐 진단** | 정상 진입/진행/이탈 및 관측 신뢰도를 진단. test 기반 phase별 threshold 최적화 금지 |
| 3 | **독립 반복·짧은 이상 및 누락 검증** | 작은 지표 차이와 평가 공백을 seed/영상 단위 불확실성 및 다른 이상 길이로 확인 |

[상세 보고서](docs/EXPERIMENT06.md) · [지표](results/experiment06/metrics.json) · [phase별 오류](results/comparison05_06/phase_errors.json) · [실험 07 계획](docs/EXPERIMENT07_PLAN.md)

## 실험 07 결과 — 다른 실제 공정 R03 적용성

R03의 정상 영상으로 로컬 Qwen discovery와 정상 모델을 새로 적합했습니다. 지게차·팔레트 공정에 R01의 단방향 ROI/공간 phase를 복사하지 않고, 같은 R03 특징에서 기본 설정과 pooled 지게차 진행량 추가를 비교했습니다. 정상 fit/calibration 18/4개, 테스트 17개 영상의 12,005프레임입니다.

| R03 지표 | 기본 외형·전이 (07) | 진행량 추가 (07_motion) |
|---|---:|---:|
| Visual AUROC | 0.6968 | 0.6968 |
| Visual AP | 0.6374 | 0.6374 |
| Process AUROC | 0.4980 | 0.5243 |
| Process AP | 0.4223 | 0.4355 |
| Combined AUROC | 0.6747 | 0.6165 |
| Combined AP | 0.5664 | 0.4942 |
| 정상 q99 기준 정상 오탐률 | 3.85% | 3.59% |
| 정상 q99 기준 이상 프레임 recall | 5.56% | 6.41% |
| 경보가 발생한 GT 이상 구간 | 10 / 17 | 11 / 17 |
| 미탐 GT 이상 구간 | 7 | 6 |

![실험 07 R03 내부 비교](results/comparison07_07_motion/comparison.png)

**의의:** R01에서 유용했던 진행량이 R03에서는 Combined AUROC -0.0582, AP -0.0722로 악화됐습니다. 모듈의 적용 조건과 실패를 확인했으며, 장면별 discovery·artifact·역할 anchor를 분리하는 구현을 추가했습니다. R01 학습 분포를 그대로 이전한 zero-shot 실험은 아닙니다.

**보완할 점:** `carrying` phase가 모든 분할에서 0개이고, 높은 역할 관측률에도 배경·부분·중복 박스가 섞입니다. 정상 정지와 왕복이 있는 공정에서 지게차 중심 속도만으로 적재 관계를 표현하기 어렵습니다. 두 설정 모두 프레임 recall이 낮고 6~7개 GT 구간을 놓쳤습니다. 서로 다른 정상 q99 기준 비교이며, 탐지 구간 집합이 달라 detected-only 지연 중앙값을 직접 개선량으로 해석하지 않습니다.

### 실험 07 이후 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보 | 검증 방향 |
|---|---|---|
| 1 | **객체 관계 기반 상태 grounding** | 지게차–팔레트 상대 위치·겹침·크기로 정상 latent state 구성. 상태 support와 탐지 품질 함께 검증 |
| 2 | **역할별 검출 품질·관측 불확실성** | 큰 배경 박스, 부분/중복 팔레트 진단. coverage를 GT recall로 해석하지 않음 |
| 3 | **공정 모듈의 조건부 사용·결합 검증** | 약한 공정 점수가 외형 신호를 희석하는지 검증. 정상 support로 규칙을 정하고 test 가중치 탐색 금지 |

[상세 보고서](docs/EXPERIMENT07.md) · [기본 지표](results/experiment07/metrics.json) · [진행량 추가 지표](results/experiment07_motion/metrics.json) · [실험 08 사전 계획](docs/EXPERIMENT08_PLAN.md)

## 실험 08 결과 — 객체 관계 기반 잠재 상태

지게차–팔레트의 상대 위치·면적비·겹침과 지게차 위치로 정상 관계 군집 4개를 적합했습니다. 실험 07의 bbox/CLIP 특징을 보존하고 phase와 그에 종속된 정상 모델만 바꿨습니다. 관계 관측률은 정상 FIT 99.81%, calibration/test 100%지만 검출·semantic phase 정확도를 뜻하지 않습니다.

| R03 지표 | 실험 07 | 실험 08 |
|---|---:|---:|
| Visual AUROC | 0.6968 | 0.7330 |
| Visual AP | 0.6374 | 0.6778 |
| Process AUROC | 0.4980 | 0.5520 |
| Process AP | 0.4223 | 0.4693 |
| Combined AUROC | 0.6747 | 0.6361 |
| Combined AP | 0.5664 | 0.5496 |
| 정상 q99 기준 정상 오탐률 | 3.85% | 3.36% |
| 정상 q99 기준 이상 프레임 recall | 5.56% | 10.36% |
| 경보가 발생한 GT 이상 구간 | 10 / 17 | 11 / 17 |

![실험 07–08 비교](results/comparison07_08/comparison.png)

**의의:** 외형 입력을 보존한 관계 phase 교체로 Visual AUROC가 +0.0362 상승했습니다. 두 branch의 개별 ranking이 올라도 고정 결합은 악화될 수 있음을 확인했습니다. 정상 calibration에서 공정 percentile 중앙값이 이전 상태별 0.190~0.964로 달라 다음 보정 실험의 근거를 확보했습니다. novelty·semantic grounding 성공·일반화가 입증된 것은 아닙니다.

**보완할 점:** Combined AUROC는 -0.0385이며 여전히 GT 이상 구간 6개를 놓쳤습니다. 상태 군집에는 독립 의미 정답이 없고, 정상 시간 순서에 따른 군집 번호와 self/next/cycle prior도 heuristic입니다. 각자의 정상 q99로 비교했습니다. 공통 탐지 9개 구간의 지연 변화 중앙값은 -104프레임이지만 신규 탐지 2개·탐지 손실 1개, 시작 전부터 경보가 켜진 구간 1개를 함께 고려해야 합니다. R03은 개발 장면입니다.

### 실험 08 이후 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보 | 검증 방향 |
|---|---|---|
| 1 | **이전 상태별 공정 점수 보정** | 정상 상태별 percentile 차이 진단에 근거. phase·외형·raw 전이·결합 고정 후 조건부 CDF만 비교 |
| 2 | **공정 결합 기여 분리** | branch 개선과 Combined 하락의 불일치 확인. Visual 단독·정상 신뢰도 결합을 test 가중치 탐색 없이 검증 |
| 3 | **잠재 상태·관측 신뢰성 검증** | 상태 support를 의미 정확도와 구분. 정상 대표 관측·bbox/관계 품질 및 순서 안정성 점검 |

[상세 결과·의의·한계·후보별 검증 기준](docs/EXPERIMENT08.md) · [지표](results/experiment08/metrics.json) · [정상 보정 진단](results/experiment08/normal_process_calibration_diagnostic.json) · [실험 09 계획](docs/EXPERIMENT09_PLAN.md)

## 실험 09 결과 — 이전 상태별 공정 점수 보정

실험 08의 특징·phase·외형 모델·전이 원점수는 보존하고, 정상 전이 점수의 보정만 이전 상태별로 나눴습니다. 정상 상태별 percentile 중앙값 범위는 0.190~0.964에서 0.438~0.491로 줄었습니다. 보정에 사용한 정상 데이터의 진단이며 독립 일반화 결과는 아닙니다.

| R03 지표 | 실험 08 | 실험 09 |
|---|---:|---:|
| Visual AUROC | 0.7330 | 0.7330 |
| Visual AP | 0.6778 | 0.6778 |
| Process AUROC | 0.5520 | 0.4627 |
| Process AP | 0.4693 | 0.4086 |
| Combined AUROC | 0.6361 | 0.7108 |
| Combined AP | 0.5496 | 0.6125 |
| 정상 q99 기준 정상 오탐률 | 3.36% | 1.85% |
| 정상 q99 기준 이상 프레임 recall | 10.36% | 2.29% |
| 경보가 발생한 GT 이상 구간 | 11 / 17 | 11 / 17 |

![실험 08–09 비교](results/comparison08_09/comparison.png)

**의의:** 보정만 바꿔 Combined AUROC +0.0747, AP +0.0629를 확인했습니다. 그러나 Process ranking은 하락했고 Visual 단독보다도 낮습니다. 정상 보정과 실제 공정 이해의 개선을 구분할 근거이며 novelty나 일반화 입증은 아닙니다.

**보완할 점:** 평균 결합에서 상태 유지 점수의 상한은 0.719~0.745로 정상 q99 0.958을 넘지 못합니다. 상태 유지 이상 4,884프레임에서 경보가 없었고 전체 recall은 2.29%입니다. 경보는 상태 변경 구간에 집중돼 그 구간의 정상 156프레임 중 128프레임도 오탐입니다. 동일한 GT 11개 구간을 탐지했으며 공통 지연 변화 중앙값은 0프레임입니다. 각 모델의 정상 q99 비교이고, R03은 개발 장면입니다.

### 실험 09 이후 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보 | 검증 방향 |
|---|---|---|
| 1 | **외형 점수를 낮추지 않는 결합** | 평균을 max로 바꾸고 Visual 단독도 비교. 상태 유지 탐지와 상태 전환 오탐의 상충 평가 |
| 2 | **체류 시간 기반 공정 신호** | 같은 self-transition 반복의 한계를 정상 머무름 길이로 보완. 정상 대기·정지 및 경계 censoring 진단 |
| 3 | **정상 보정·상태 관측의 안정성 검증** | calibration 4개 영상·희소 상태 16개 전이의 한계 점검. 영상 단위 support와 의미 정확도 구분 |

[상세 결과·의의·한계·후보별 검증 기준](docs/EXPERIMENT09.md) · [지표](results/experiment09/metrics.json) · [결합 실패 진단](results/experiment09/fusion_diagnostic.json) · [실험 10 계획](docs/EXPERIMENT10_PLAN.md)

## 실험 10 결과 — 최댓값 결합과 Visual 단독

두 branch를 고정하고 평균 결합을 max로 교체했습니다. Visual 단독 대조군도 함께 평가했으며, 각 설정의 최종 점수로 정상 q99를 따로 정했습니다.

| R03 최종 점수 지표 | 09 평균 | 10 최댓값 | Visual 단독 |
|---|---:|---:|---:|
| AUROC | 0.7108 | 0.7339 | 0.7330 |
| AP | 0.6125 | 0.6784 | 0.6778 |
| 정상 q99 기준 정상 오탐률 | 1.85% | 2.62% | 7.87% |
| 정상 q99 기준 이상 프레임 recall | 2.29% | 25.59% | 31.53% |
| 경보가 발생한 GT 이상 구간 | 11 / 17 | 11 / 17 | 13 / 17 |

![실험 10 결합 비교](results/comparison09_10_10_visual/comparison.png)

**의의:** 평균 결합에서 경보가 불가능했던 상태 유지 이상 4,884프레임 중 max는 1,213프레임을 탐지했습니다. 결합의 억제는 해소됐지만 Visual 대비 AUROC 차이는 +0.0009이며 추가 탐지 GT 구간은 없었습니다. 공정 신호의 추가 기여가 제한적임을 확인한 결과이고 novelty·일반화 입증은 아닙니다.

**보완할 점:** max도 이상 구간 6개를 놓쳤습니다. Visual보다 높은 q99 때문에 기존 이상 경보 317프레임을 잃고 16프레임을 추가했으며, 구간 2개를 더 놓쳤습니다. 점수가 낮아지지 않아도 재보정한 임계값의 탐지가 보존되는 것은 아닙니다. 각자의 정상 q99 비교이며 작은 ranking 차이에 유의성을 주장하지 않습니다.

### 실험 10 이후 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보 | 검증 방향 |
|---|---|---|
| 1 | **관측된 진입 이후 정상 체류 시간** | 전이만으로 부족한 지속 신호 추가. 정상 support가 충분한 상태만 사용하고 경계·누락·진입 미관측은 abstain |
| 2 | **영상 단위 정상 보정 안정성** | q99 차이 0.0008219에 경보가 크게 변함. 정상 영상별 상단 분포와 임계값 민감도 점검 |
| 3 | **관계 상태의 의미·관측 품질** | 짧은 군집 전환이 실제 동작인지 확인. 검출 실패·jitter와 실제 전환을 보조 진단으로 구분 |

[상세 결과·의의·한계·후보별 검증 기준](docs/EXPERIMENT10.md) · [최댓값 지표](results/experiment10/metrics.json) · [Visual 지표](results/experiment10_visual/metrics.json) · [경보·상단 점수 진단](results/comparison09_10_10_visual/fusion_diagnostic.json) · [실험 11 계획](docs/EXPERIMENT11_PLAN.md)

## 실험 11 결과 — 관측된 진입 이후 체류 시간

정상 완결 구간이 충분한 상태 1/2/3에만 체류 점수를 추가했습니다. 시작 경계·관계 누락·지원 부족 상태에서는 기존 전이를 유지합니다. 특징·phase·Visual·기존 전이 점수는 보존했습니다.

| R03 지표 | 실험 10 | 실험 11 |
|---|---:|---:|
| Visual AUROC | 0.7330 | 0.7330 |
| Process AUROC | 0.4627 | 0.5230 |
| Combined AUROC | 0.7339 | 0.7005 |
| Combined AP | 0.6784 | 0.6055 |
| 정상 q99 기준 정상 오탐률 | 2.62% | 6.37% |
| 정상 q99 기준 이상 프레임 recall | 25.59% | 26.22% |
| 경보가 발생한 GT 이상 구간 | 11 / 17 | 11 / 17 |

![실험 10–11 비교](results/comparison10_11/comparison.png)

**의의:** 경계·누락·지원 부족을 구분하는 인과적 체류 모듈을 구현하고 실패 조건을 확인했습니다. Process ranking은 조금 높아져도 최종 성능은 악화됐으며, 현재 결과로 체류 모듈의 성능 개선이나 novelty를 주장할 수 없습니다.

**보완할 점:** 정상 오탐 260프레임과 이상 탐지 32프레임이 추가됐고 새 GT 구간은 탐지하지 못했습니다. 상태 2에서 추가 이상 탐지 없이 오탐 224프레임이 발생했습니다. 체류 신호는 테스트 프레임 58.43%에서만 유효하며, 유효 구간 단독 AUROC는 0.5050입니다. 정상 길이도 진입 맥락에 따라 달랐지만 이것이 실패 원인인지는 후속 검증이 필요합니다. 각자의 정상 q99 비교이며 R03은 개발 장면입니다.

### 실험 11 이후 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보 | 검증 방향 |
|---|---|---|
| 1 | **진입 맥락별 체류 시간** | 상태 2의 정상 길이 중앙값 134 대 18프레임 차이 검증. 동일 유효 마스크 대조군으로 abstention 효과와 분리 |
| 2 | **정상 속도·상단 보정 안정성** | 체류 percentile=1인 테스트 정상 284프레임. 정상 영상별 길이 꼬리·calibration 대표성 점검 |
| 3 | **관계 상태와 실제 동작 대응** | 의미 GT 부재와 약한 체류 구별력 점검. 군집 재방문·jitter·가림과 실제 지속 구분 |

[상세 결과·의의·한계·후보별 검증 기준](docs/EXPERIMENT11.md) · [지표](results/experiment11/metrics.json) · [체류 오류 진단](results/experiment11/dwell_diagnostic.json) · [정상 진입 맥락](results/experiment11/normal_entry_context_diagnostic.json) · [실험 12 계획](docs/EXPERIMENT12_PLAN.md)

## 로컬 VLM

| 설정 | 값 |
|---|---|
| 실행기 | `/home/jeong/Server/llama.cpp/build/bin/llama-server` |
| 모델 | Qwen3.8-27B-Q8_0 + 대응 mmproj |
| context | 요청당 65,536 |
| temperature | 0.2 |
| GPU layers | all |
| reasoning | on |
| parallel | 1 (요청당 context 확보를 위한 사용자 승인) |
| 나머지 서버 옵션 | 기본값 |

`bash scripts/start_local_vlm.sh`로 실행합니다. OpenAI 호환 HTTP 형식을 사용하는 **로컬 loopback endpoint**이며 외부 유료 API가 아닙니다. 생성된 reasoning 텍스트는 결과 파일에 저장하지 않습니다.

## 출처

- [IPAD 논문](https://arxiv.org/abs/2404.15033), [공식 코드 고정 commit](https://github.com/LJF1113/IPAD/tree/22764cbeeda3946303d236babdd2664fd6241b91)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [CLIP](https://github.com/openai/CLIP)
- [SubspaceAD](https://github.com/CLendering/SubspaceAD): 정상 subspace residual이라는 발상 참고. 본 구현은 CLIP crop 특징을 사용하며 원 논문의 DINOv2 patch-level 구현과 동일하지 않습니다.
