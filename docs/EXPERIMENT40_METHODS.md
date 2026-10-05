# 실험40 구현·재현 명세

이 문서는 비교 구현과 재현 절차다. 완료 여부와 결과 해석은 `EXPERIMENT40.md` 및 `results/experiment40/`의 실제 실행 기록으로 판단한다. `EXPERIMENT40_PLAN.md`는 실행 전 동결한 기록이므로 완료 후에도 덮어쓰지 않는다.

## 학습 범위

공통 visual tower 하나를 R01~R04의 정상 영상으로 학습한다. 이번에 학습한 것은 detector·ReID·phase head가 아니다. 네 공정이 시각 encoder 가중치를 공유한다는 구현을 검증하며, 공정별 encoder 대비 공유 학습의 우월성이나 공정 불변 표현을 입증한 실험은 아니다.

| arm | 구현 | 전처리 | 학습 가능한 파라미터 |
|---|---|---|---:|
| A | 기존 OpenAI CLIP ViT-B/32 | shortest-edge224, bicubic, center crop224, CLIP mean/std | 0 |
| B | MobileCLIP2-S2 공식 OpenCLIP/timm 변환 | shortest-edge256, bilinear, center crop256, RGB [0,1] | 0 |
| C | B의 attention qkv/proj 8개 + 출력 projection 1개에 rank8/alpha16 LoRA | B와 동일 | 137,216 |
| D | B의 visual tower 전체 FT | B와 동일 | 35,815,232 |

C의 `W(x) + (alpha/rank) B(A(x))`에서 B는 0 초기화다. 원본 W는 고정한다. D는 conv/linear/normalization affine 등 전체 visual parameter를 최적화한다. 두 학습 arm 모두 BN running statistics 및 dropout 동작은 평가 모드로 고정한다. 따라서 D를 BN 통계까지 재추정하는 full-training-mode 실험으로 해석하면 안 된다. Text tower는 로드·학습하지 않는다. C 체크포인트는 재현 편의를 위해 전체 visual state를 저장하므로 파일 크기를 adapter-only 배포 크기로 주장하지 않는다.

## 데이터와 목적함수

정상111영상 중 기존 downstream FIT89영상을 표현 train70/validation19로 영상 단위 분할했다. 정상 calibration22영상은 encoder 최적화와 epoch 선택에서 제외했다. 각 epoch4096 views, batch64, 네 공정 각16 views, 각 공정 안에서 global/crop 각8 views를 사용한다. 영상 노출을 순환·무작위화하며 동일 seed의 C/D가 같은 view/augmentation을 받는다. 전체 normal manifest54,047 views를 매 epoch 모두 학습하는 방식은 아니다.

두 작은 crop/color augmentation view를 같은 instance의 positive로 사용한다. negative는 **같은 공정의 다른 영상**으로 제한한다. 다른 공정과 같은 영상의 다른 view는 negative에서 제외한다. 그럼에도 다른 영상의 동일 정상 상태가 false negative가 될 수 있다. Track ID·phase·anomaly label은 학습 정답으로 사용하지 않는다.

L = L_teacher + 0.1 L_contrastive. L_teacher는 두 augmentation embedding과 canonical frozen-B embedding의 평균 cosine distance다. L_contrastive는 두 view 사이 양방향 cross-entropy이며 temperature0.1이다. Teacher gradient는 차단한다. 이는 기존 자기지도·표현 보존 기법의 조합 구현으로, 새로운 loss의 이론적 novelty를 주장하지 않는다.

C/D 공통: seeds42/43/44, 10epochs×64steps=640 updates, AdamW lr3e-5/weight_decay0.01, warmup1epoch 후 cosine decay, gradient clipping1.0, BF16 forward/FP32 optimizer, deterministic cuDNN, TF32 off. Epoch0을 포함한 고정 정상 validation512 views의 최소 objective로 checkpoint를 선택한다. 동일 예산·학습률 비교이며 각 방법의 최적 hyperparameter를 탐색한 결과는 아니다. Train objective와 validation objective는 샘플/augmentation 모집단이 달라 차이를 곧바로 과적합 크기로 해석할 수 없다.

## 공정별 정상 모델

| 공정 | 고정한 phase·process branch | 결합 |
|---|---|---|
| R01 | 실험02 공간 phase, 기존 transition grammar | Visual/Process 평균 |
| R02 | 정상 FIT 로컬 Qwen vocabulary/grammar + frozen CLIP phase | Visual/Process 평균 |
| R03 | 실험14 관계 phase + entry complete-percentile dwell | max |
| R04 | 실험39 hold: 실험37 asinh phase + 동일 pair 진입 근거 gated lognormal dwell | max |

R02는 기존 캐시가 없어 새 frozen 기준 캐시를 만들었다. 정상 FIT02/03에서 총12프레임을 로컬 Qwen으로 해석했으며 추정한 단계는 phase 정답이 아니다. 네 공정에서 downstream 알고리즘이 동일하다고 주장하지 않는다. HSMM 신경망이나 공정별 LoRA 순서 모델은 이번 실험에 추가하지 않았다.

A~D는 동일 source-frame indices/boxes/roles/tracks/phases를 공유한다. A도 같은 crop 정의로 시각 특징을 재추출한다. 모든 비시각 배열이 동일한지 검사한다. R02의 CLIP phase 및 R04의 CLIP anchor 선택을 다시 MobileCLIP으로 계산하지 않는다. 그러므로 MobileCLIP만 사용하는 실시간 end-to-end 시스템의 비용 비교가 아니다.

정상 appearance PCA는 공정/arm별로 기존 FIT89 중 해당 공정 자료에 다시 적합한다. 공통 규칙은 variance0.95, max_rank32, minimum_samples10이다. 실제 rank와 phase 지원 부족 fallback은 기록한다. R04의 이전 수동 rank 고정값은 모든 arm에서 제거했으므로 A가 실험39와 완전히 같은 예측일 필요는 없다.

Residual은 중심화 특징의 PCA 직교 여공간 제곱거리다. 정상 calibration CDF로 변환한 객체/전체 프레임 점수의 최댓값을 Visual로 사용한다. 고정된 공정 branch와 결합한 뒤 각 공정/arm의 정상 calibration q99를 사용하며 경보 조건은 strict `score > q99`다. 별도 normal-video leave-one-out calibration 검증으로 미사용 정상 영상의 오탐도 확인한다. 공정 파라미터·점수의 arm 간 동일성을 검사한다.

## 평가와 실행 기록

표현 학습 protocol → 정상 모델/holdout/임계값 checkpoint → test 특징 → 모든 test 점수 checkpoint → label 평가 순서를 따른다. 원본 label 길이가 맞지 않는 영상은 전체 -1로 제외한다. 점수는 source-frame index에 인과적 zero-order hold로 확장한다. FPS·초 단위 지연·cycle 경계를 만들어내지 않는다.

공정 안의 Visual/Combined AUROC/AP, 개별 q99 FPR/recall, GT 이상 구간 중 한 번이라도 경보가 발생한 구간 비율을 보고한다. 공정 macro는 네 공정의 지표를 동일 가중 평균한 값이며 raw score를 합쳐 새 AUROC를 계산한 값이 아니다. C/D mean±sample SD는 같은 영상 분할에서 세 학습 seed(순서/augmentation 및 LoRA 초기화)의 기술 통계다. 세 독립 데이터 분할이나 독립 데이터셋에 대한 신뢰구간이 아니다. 구간 탐지는 bbox localization 정확도나 action-phase 정확도가 아니다.

초기 GPU 수치 smoke에서 전체 tower의 bitwise 비교가 실패해 float32 허용오차를 명시했다. 최종 smoke는 실제 gradient/update·LoRA base 보존·BN buffer 보존·복원을 확인했다. 첫 DataLoader 시도는 호스트 메모리 압박으로 중단하고 worker만 줄여 원본 pretrained에서 재시작했다. `resource_preflight_recovery.json`에 중단 범위·변경·보존 위치가 있다. 최종 아카이브는 epoch1·64updates와 epoch1 checkpoint를 포함한다. 동결 계획의 “완료 epoch 없음”은 중단 요청 당시의 초기 관찰이었으며 최종 종료 상태와 달라 복구 기록에서 정정했다. 중단 시도의 진행은 모두 폐기했고 완료 결과에 섞지 않는다. 재시작 후 epoch1 train/validation 수치는 중단 시도의 epoch1과 정확히 일치했다.

## 재현 명령

저장소 루트에서 실행한다. 기존 Stage00 split, 이전 실험의 정상 phase/검출 캐시와 pinned CLIP/GroundingDINO 모델이 필요하다. 원본 데이터·모델·feature cache는 저장소에 포함하지 않는다. 기존 결과 디렉터리에 덮어쓰지 않고 별도 재현 checkout을 사용한다. 새 checkout에서도 공개된 `results/experiment40/`가 존재하므로 먼저 그 디렉터리를 `results/experiment40_recorded/` 등으로 이동해 원본 기록을 보존한다. 실행 스크립트의 기본 출력은 `results/experiment40/`와 ignored `artifacts/experiment40/`로 고정되어 있다. 기존 로컬 artifacts를 새 출력 위치에 섞지 않는다. 아래는 성공한 실행 순서이며 중단 시에는 해당 stage/log를 검토한 뒤 복구해야 한다.

```bash
uv pip install --python .venv/bin/python --target artifacts/experiment40/deps --no-deps -r requirements-experiment40.txt
.venv/bin/python scripts/download_representation40_model.py
export PYTHONPATH=artifacts/experiment40/deps:src
.venv/bin/python scripts/extract_features.py --data-root /media/jeong/ExtHDD/IPAD_vLLM/IPAD_dataset --config configs/experiment40_R02_base.json --partition training
.venv/bin/python scripts/prepare_representation40.py
.venv/bin/python scripts/smoke_representation40.py
.venv/bin/python scripts/extract_representation40.py --arm A
.venv/bin/python scripts/extract_representation40.py --arm B
.venv/bin/python scripts/freeze_representation40.py
.venv/bin/python scripts/run_representation40_training.py
.venv/bin/python scripts/run_representation40_evaluation.py
.venv/bin/python scripts/validate_representation40.py
.venv/bin/python scripts/diagnose_representation40.py
.venv/bin/python scripts/tabulate_representation40.py
.venv/bin/python scripts/plot_representation40.py
.venv/bin/python -m pytest -q
```

학습 GPU peak는 PyTorch allocated memory이며, VLM·detector를 동시에 실행하는 전체 파이프라인 메모리가 아니다. Teacher feature는 미리 추출해 두었고 학습 중 teacher tower를 GPU에 함께 올리지 않았다. 실제 실행은 A/B 정상 추출 순서와 GPU smoke 일부가 겹쳤으므로 extraction wall time은 고립된 성능 benchmark가 아니다. `run_representation40_evaluation.py`는 이미 실행 중인 학습 queue가 완료되기를 기다린 뒤 순차 평가할 수도 있다. 재현 환경에서는 공개된 결과/protocol을 원본 기록으로 보관하고 새 실행 출력 경로를 분리해야 한다. 원본 protocol을 지우고 과거 결과인 것처럼 덮어쓰면 안 된다.

모델 revision `ac6b37c8fc40b62b623d09ed228a6f0c9bc29fe6`: [공식 OpenCLIP/timm 변환](https://huggingface.co/timm/MobileCLIP2-S2-OpenCLIP). 구현 출처: [Apple MobileCLIP](https://github.com/apple-aiml-research/ml-mobileclip), [OpenCLIP](https://github.com/mlfoundations/open_clip). 실행 패키지 버전은 `results/experiment40/environment.json`에 있다.
