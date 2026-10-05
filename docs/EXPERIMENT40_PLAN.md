# 실험 40 계획 — 공정 공유 Learned Visual Representation

상태: 사용자 지시에 따라 기존 체류 학습 계획을 대체했다. 이전 안은 [보류 기록](EXPERIMENT40_DWELL_DEFERRED.md)에 보존한다. 실험39까지의 결과/코드는 변경하지 않는다. 정상111영상/54,047 views 준비와 GPU smoke를 완료했다. encoder 학습·테스트 성능 결과는 아직 없다.

## 범위와 질문

R01/R02/R03/R04의 공통 시각 표현은 공유하고, phase/transition/dwell/정상 appearance 모델/calibration은 공정별로 분리한다. 이번 비교에서는 visual encoder만 변경한다. GroundingDINO 학습, learned ReID, learned phase head는 후속 단계에서 각각 검증하며 이번에 함께 학습하지 않는다. HSMM/transition/dwell을 LoRA 대상이라고 간주하지 않는다. LoRA는 신경망 encoder에 적용한다.

| 구성 | 시각 표현 | 업데이트 범위 |
|---|---|---|
| 40-A | 기존 CLIP ViT-B/32 | frozen |
| 40-B | MobileCLIP2-S2 | frozen |
| 40-C | 동일 MobileCLIP2-S2 | attention qkv/proj 및 시각 출력 projection에 LoRA, rank8/alpha16 |
| 40-D | 동일 MobileCLIP2-S2 | 시각 tower 전체 파라미터 FT |

MobileCLIP2는 공식 OpenCLIP/timm 변환 체크포인트를 revision 고정으로 사용한다. 텍스트 tower는 학습하지 않는다. BatchNorm running statistics는 C/D 모두 고정하고 D의 affine/conv/linear 등 모든 시각 파라미터는 학습한다. 실제 학습 가능 파라미터는 C 137,216개(추가 LoRA), D 35,815,232개(전체 visual)다. C는 attention qkv/proj 8개와 출력 projection 1개에 적용한다. 부분 FT를 추가로 섞지 않고 이번 C는 LoRA 하나로 고정한다.

## 비교 통제

- R01 기존 공간 phase/검출 캐시, R03 기존 관계 phase/검출 캐시, R04 실험39의 고정 asinh phase/검출 캐시를 재사용한다. R02는 기존 캐시가 없으므로 정상 FIT의 로컬 Qwen vocabulary와 frozen GroundingDINO/CLIP 기준 캐시를 먼저 만든다. R02의 기준 phase는 frozen CLIP 상태 추정으로 고정한다. 공정별 phase 추정 방식이 같다는 주장은 하지 않는다.
- A~D의 입력 영상/샘플 간격/boxes/roles/tracks/phase는 동일하다. 다른 encoder를 사용했다고 객체나 phase를 다시 선택하지 않는다. R04의 CLIP 기반 anchor 선택도 기존 결과로 고정한다. 이 controlled-cache 비교는 MobileCLIP 단독 실시간 배포/속도의 검증이 아니다.
- 동일 bbox를 원본 영상에 적용한 crop과 전체 frame을 모두 사용한다. A도 동일 이미지 경로로 CLIP을 재추출한다. 각 encoder의 공식 입력 해상도/전처리를 사용하고 차이를 기록한다.
- 정상 appearance PCA/CDF는 표현이 바뀌므로 각 arm·공정에서 다시 적합하되 같은 정상 분할·설정 규칙을 사용한다. PCA는 모두 variance.95/max_rank32/minimum_samples10의 같은 규칙을 적용한다. 실제 rank는 표현에 따라 달라질 수 있어 공개하고, R04의 예전 실험별 rank 강제값은 이번 A~D 전체에 적용하지 않는다. 따라서40-A가 실험39의 최종 예측을 bitwise 재현한다고 주장하지 않는다.
- 공정 branch는 R01의 기존 spatial baseline(02), R02의 새 frozen VLM/CLIP phase baseline, R03의 기존 entry complete-percentile dwell(14), R04의 기존 same-pair gated lognormal dwell(39 hold)을 공정별로 유지한다. R01/R02에 이번에 새 dwell/HSMM을 도입하지 않는다. 네 공정이 동일한 downstream 모델이라는 주장은 하지 않고, encoder 변화는 각 공정 안에서 비교한다. 모든 arm에서 공정 입력과 파라미터를 대조한다.

## 정상-only 표현 학습

- 기존 공정별 FIT89영상을 representation-train70/representation-validation19로 **영상 단위** 분할했다. calibration22영상은 별도로 유지한다. calibration 영상과 test는 encoder 학습·checkpoint 선택에서 제외한다. 기존 FIT 전체는 downstream 정상 모델 적합에만 다시 쓸 수 있다. 독립 녹화 그룹이 알려져 있지 않아 영상 분할을 완전한 독립성으로 주장하지 않는다.
- 네 공정, 각 영상 및 global/crop의 노출을 균형 있게 샘플링한다. 두 augmentation view의 instance 대조 목적과 frozen MobileCLIP2 teacher에 대한 cosine 보존 항을 함께 사용한다. 이는 추적 ID나 phase 정답을 학습 목표로 쓰는 ReID/phase 학습이 아니다.
- 작은 crop/color 변화만 쓰고 좌우 반전·시간 순서 변형·임의 큰 회전은 쓰지 않는다. 같은 영상의 다른 sample을 negative에서 제외하고 동일 공정 내 다른 영상 negative를 사용한다. 여전히 서로 다른 정상 영상의 같은 상태가 false negative일 수 있음을 기록한다.
- C/D는 같은 데이터·augmentation·loss·batch·optimizer·학습률·업데이트 예산으로 비교한다. 초기 실행안은 seeds42/43/44, batch64, 10epochs, epoch당4096 samples, AdamW lr3e-5/weight_decay.01, LoRA rank8/alpha16이다. 동일 학습률 비교는 각 방법의 최적 hyperparameter 탐색이 아니다. 실행 전 정상 smoke 검증으로 메모리/수치 안정성을 확인하며 변경하면 동결 이전에 기록한다.
- 정상 representation-validation objective로만 checkpoint를 선택한다. epoch0도 후보에 포함하며 학습 checkpoint가 선택되지 않는 결과를 숨기지 않는다. 테스트 AUROC로 epoch/seed/arm을 고르지 않는다.

## 검증·평가

1. 데이터 분할·원본 feature 해시·다운로드 revision·코드·설정·학습 가능한 parameter 목록을 실행 전에 동결한다. 정상 접근 allowlist로 test/label을 차단한다.
2. LoRA zero-init의 frozen 출력 일치(Linear 단위는 exact, GPU 전체 tower는 rtol1e-5/atol1e-6), 동결 parameter/buffer 불변, 실제 C/D gradient와 업데이트, 체크포인트 복원, train/val/calibration 분리, 같은 입력 재사용을 검증한다. 랜덤 초기화나 frozen으로 끝난 실행을 학습 성공으로 보고하지 않는다.
3. 각 seed의 train/validation loss·best epoch·파라미터 수·wall time·GPU peak memory·teacher drift·feature 분산/유효 rank를 기록한다. loss 감소를 anomaly 성능 향상으로 대신하지 않는다.
4. encoder를 고정한 뒤 각 공정의 normal appearance model과 calibration을 적합하고 정상 holdout을 검증한다. 그 결과와 임계값을 동결한 다음 test를 평가한다.
5. 공정별 Visual/Combined AUROC/AP·정상 q99 FPR/recall·구간 탐지·표현 붕괴 및 공정별 퇴행을 보고한다. A→B는 architecture/pretraining 차이, B→C/D는 adaptation, C↔D는 동일 예산의 학습 범위 비교로 구분한다. 서로 다른 공정의 raw score를 섞은 단일 AUROC를 주 결론으로 쓰지 않는다.
6. 기존 라벨 길이 불일치는 -1 제외 정책을 유지한다. 원본 라벨을 추정 보정하거나 FPS를 만들어내지 않는다. LoRA≥full FT는 검증할 가설이며 결과 방향을 전제하지 않는다.

## 후속 진행

완료하면 결과·의의·보완점·추천순3개를 보고서/README/그래프로 정리하고 GitHub에 업로드한다. 41번부터 GroundingDINO training → learned ReID → learned phase head의 큰 방향을 유지하되, 직전 결과에 근거해 다음 변경 하나만 구체화한다. 이번 단계에서 전체 후속 실험을 확정하지 않는다. 학부 캡스톤의 공유 시각 인식/공정별 정상 모델 파이프라인 구현을 중심으로 두고 성능만으로 novelty를 주장하지 않는다.

## 수치 사전검증 기록

GPU의 bitwise tower 비교는 기본 TF32에서 최대3.22e-5, TF32 비활성에서8.94e-8 차이로 실패했다. 학습은 deterministic cuDNN/TF32 off로 고정하고, GPU float32 전체 출력 검증은 rtol1e-5/atol1e-6로 명시했다. 최종 smoke에서 C 초기 출력 차이0, D 8.94e-8이며, 복사/반복 대조는0이다. C/D 실제 gradient·parameter update·BN buffer 보존을 확인했다. 합성 smoke 가중치는 모두 폐기하고 학습에는 원본 pretrained를 다시 로드한다. 이것은 학습 성능 결과가 아니다.

## 실행 자원 사전조정

첫 C_s42 시도는 epoch0 평가 후 학습 DataLoader 대기 중 호스트 RAM58/62GiB·swap 포화가 확인되어 중단했다. 완료된 학습 epoch는 없으며 부분 optimizer 진행은 결과로 사용하지 않는다. 해당 protocol/epoch0 기록과 checkpoint를 resource_preflight_attempt1로 보존했다. 학습 worker4→2, 검증 worker4→0, prefetch1로 바꾸고 동일 seed·학습 샘플/augmentation·loss·학습률·예산으로 원본 pretrained에서 다시 시작한다. 테스트 결과에 따른 변경이 아니다.

## 모델 출처

- [Apple MobileCLIP2 공식 구현](https://github.com/apple-aiml-research/ml-mobileclip)
- [MobileCLIP2-S2 공식 가중치](https://huggingface.co/apple/MobileCLIP2-S2)
- [OpenCLIP/timm 공식 변환](https://huggingface.co/timm/MobileCLIP2-S2-OpenCLIP)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
