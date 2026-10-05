# 실험45 — 더 큰 visual encoder의 통제 비교와 후보 선정

상태: 실행 전 계획. 사용자의 새 목표에 따라 모델 크기·학습·특징 수·시간 단위를 순차 실험한다. 독립 자료가 필요한44는 미실행 상태로 보존한다. 이후 실험 번호와 구체적 하이퍼파라미터는 이번 결과에서 정한다.

## 질문과 범위

40의 MobileCLIP2 LoRA는 full FT보다 좋았지만,41~43에서 정상 loss/weak agreement 개선이 AD 개선으로 연결되지 않았다. 이번에는 detector/box/tracks/phase/process score를40과 동일하게 유지하고 visual encoder만 교체한다. 더 큰 모델이라는 이유만으로 우수하다고 가정하지 않는다.

실제 비교 후보는 공식 checkpoint의 CLIP ViT-L/14, SigLIP2-L/16-384, DINOv2-L/14이다. 모델 revision은 `configs/experiment45_candidates.json`에 고정한다. 각 모델의 공식 image processor와 vision representation을 사용하며 입력 해상도·학습 데이터·사전학습 목적·출력 차원이 다르므로 순수한 parameter scaling 실험으로 해석하지 않는다. Text tower나 원격 코드는 실행하지 않는다. CLIP은 projected pooled feature, SigLIP은 vision pooler, DINOv2는 CLS를 사용한다. 모두 L2 normalize한다.

40의 A(frozen CLIP-B), B(frozen MobileCLIP2-S2), C 3seed(LoRA)는 기존 결과/캐시를 검증한 대조다. 대형 후보3개를 모두 정상111 및 개발 test66영상에서 실제 추론하고 같은 PCA(.95, rank cap32)/공정 모델/CDF/q99로 평가한다. 신규 학습은 다음 단계에서 선택 모델에 수행한다. 이번 단계는 frozen candidate screening이다.

## 라벨을 열기 전 선택 기준

기존 video split을 보존한다: 정상 train70 / representation validation19 / calibration22. 후보 선택에는 train70의 특징과 val19의 원본·약한 밝기 변화·합성 국소 결함만 사용한다. Calibration 및 test는 선택에서 제외한다. 정상만으로 실제 anomaly recall을 최적화할 수 없으므로 합성 probe는 가설 선택용 proxy이며 실제 결함 정답이 아니다.

Scene×global/crop별 video-balanced 결정론적 train512, val64 view를 선정한다(총 train4096,val512). Train 원본으로 PCA를 적합한다. Val 원본 및 brightness1.05를 negative, 면적16% 국소 occlusion/4분할 shuffle/noise 각각을 positive로 하여 PCA residual AUROC를 계산한다. Scene×view kind×corruption 24개 AUROC 평균으로 대형 후보를 선정한다. 최고값과0.005이내인 후보는 같은 probe에서 측정한 encoder inference latency가 낮은 것을 선택한다. 합성 proxy 순위와 실제 AD 순위가 다르면 그대로 보고하며 test로 선정 모델을 바꾸지 않는다. 모든 corruption은 모델별로 같은 이미지·위치·난수다.

## 고정 및 검증

- 학습·검증 영상 분할/원본 이미지 view manifest/source NPZ/checkpoint/코드 hash를 보존한다. 정상 fitting 중 test 접근, test scoring 완료 전 label 접근을 차단한다.
- Detector/phase metadata가40과 완전히 같음을 검증한다. Global/crop feature 차원 변경만 허용한다. 동일 공정 score와 전이/dwell parameter를 검증한다.
- 각 모델/process별 정상 calibration q99, 정상 calibration video holdout FPR, 개발 AD AUROC/AP/FPR/recall/event coverage 및 영상별 변화를 보고한다. 별도 threshold임을 밝힌다.
- 입력 크기, 실제 visual parameter 수, 차원, 유한성/L2 norm, 처리 비용/peak VRAM, normal PCA rank/support를 기록한다. 비용은 측정 범위만 주장한다.
- R02 라벨 길이 불일치1,912프레임은 기존 strict unknown 정책을 유지한다. FPS를 추정하지 않는다. 반복 test는 개발 평가이며 독립 일반화/SOTA 근거가 아니다.
- 완료 후 결과·의의·보완점·추천순3개, README, 재현 코드/설정/수치/그래프를 검증하고 GitHub에 업로드한다. 결과에 따라 선택 대형 모델의 LoRA/partial/full FT 및 충분한 학습·early stopping 설계를 다음 실험에서 구체화한다. 특징 수/시간 단위 변경을 아직 같이 적용하지 않는다.

## 공식 근거

- [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14): CLIP 계열을 유지한 큰 visual tower 대조.
- [SigLIP2-L](https://huggingface.co/google/siglip2-large-patch16-384): 언어 정렬과 local/dense feature 사전학습 후보.
- [DINOv2 공식 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md), [HF checkpoint](https://huggingface.co/facebook/dinov2-large): self-supervised visual representation 후보. 기존 text phase descriptor는 기존 CLIP 경로로 고정한다.
