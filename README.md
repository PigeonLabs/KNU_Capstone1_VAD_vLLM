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
| 다음 실험 05 | 여러 연속 관측의 진행량 | 계획 완료, 미실행 | [계획](docs/EXPERIMENT05_PLAN.md) |

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

**실험 02의 단일 변경:** detector, crop 특징, encoder, split, sampling, PCA/scoring 규칙은 그대로 유지하고 phase 추정만 바꿉니다. 정상 FIT에서 실제로 이동한 product-role track으로 이동 경로와 3개 공간 상태를 적합하고, test-time에는 현재 bbox와 과거 상태만 사용합니다. 누락 시 이전 phase를 유지하며 그 비율을 공개합니다. 위치 정답·test label·미래 프레임은 사용하지 않습니다. 이는 R01의 공간적 진행 proxy이며 일반 공정 grammar 학습으로 주장하지 않습니다. 그 이후 실험은 아직 설계하지 않습니다.

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

[상세 보고서](docs/EXPERIMENT04.md) · [지표](results/experiment04/metrics.json) · [오류 진단](results/experiment04/error_diagnosis.json) · [다음 실험 05 계획](docs/EXPERIMENT05_PLAN.md)

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
