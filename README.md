# 산업 공정 영상 이상탐지 파이프라인

정상 영상으로 공정 패턴을 학습하고 객체별 외형·공정 흐름의 이상을 찾는 학부 캡스톤 연구입니다. 특정 논문 주제를 미리 고정하지 않고, 기본 파이프라인의 관측 결과를 바탕으로 다음 실험을 설계합니다.

## 진행 현황

| 단계 | 범위 | 상태 | 결과 |
|---|---|---|---|
| 데이터 정리 (Stage 00) | IPAD 16개 장면 전체 | 안전한 로더·평가 마스크 완료, 정확한 재라벨링은 미해결 | 아래 표 및 [기계 판독 결과](results/stage00/summary.json) |
| 실험 01 | R01 전체 학습/테스트를 사용하는 기본 파이프라인 pilot | 준비 중; 성능 미측정 | [고정 설정](configs/experiment01.json) |
| 실험 02 | 미정 | 실험 01 결과 검토 후 결정 | 사전 실험 계획 없음 |

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
