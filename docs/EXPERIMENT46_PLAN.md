# 실험46 계획 — 선정한 CLIP-L의 LoRA / partial FT / full FT

상태:45 완료 결과에서 선택한 다음 학습 실험. 이 문서 작성 시46 학습 결과는 아직 없다.46에서는 visual adaptation 한 요소만 바꾸며 rank/검출 수/시간 window를 동시에 바꾸지 않는다.

## 45에서 선택한 근거

Normal-only synthetic proxy로 고정한 후보는 CLIP-L이다(AUROC.9909). 실제 개발 AD에서는 frozen CLIP-L의 Combined AUROC.6391/recall13.68%가 MobileCLIP frozen의.6584/16.29%보다 낮았다. 단순한 모델 확대만으로 개선되지 않았으므로 정상 공정 자료에 대한 적응이 필요한지를 검증한다. Proxy 선택과 실제 AD 결과가 달랐음을 숨기거나 test로 후보를 교체하지 않는다.

CLIP-L의70개 정상 subspace 중63개가 rank32 상한에서 분산 목표95%에 미달했고 평균 보존율은90.16%였다. 이는 중요한 후속 후보지만,46에서는 rank를 그대로 두어 학습 효과와 용량 변경을 분리한다.40의 MobileCLIP에서는 LoRA가 full FT보다 좋았으나 큰 CLIP-L에서도 같을지는 미확인이다.

## 비교군과 변경 범위

| 군 | 실제 변경 |
|---|---|
| F |45의 frozen CLIP-L feature/정상 모델 대조 |
| L |24개 vision Transformer attention의 q/k/v/out linear에 rank8, alpha16 LoRA. 원본 weight/projection/norm은 frozen |
| P |마지막2개 vision Transformer block + 최종 layer norm + visual projection만 partial FT |
| D |text tower를 제외한 visual tower·projection 전체 full FT |

L/P/D 각각 seed42/43/44를 실제 학습한다. F는 신규 학습이 없으며 기존 결과의 exact replay로 검증한다. GPU smoke에서 weight load, zero-init LoRA의 원본 출력 유지, 각 모드의 trainable parameter 목록·수, gradient와 실제 update를 확인하고 본 학습 전에 protocol을 고정한다. 선언한 layer가 match하지 않으면 조용히 다른 대상을 학습하지 않고 중단한다.

## 정상 학습·검증과 충분한 학습

- Split은70 representation train /19 validation /22 calibration 그대로다. Encoder 학습·checkpoint 선택에서 calibration/test는 제외한다. Dataset allowlist와 audit hook, split/source hash로 검증한다. 기존 관측 전처리의 과거 validation 노출 한계는45와 동일하다.
- Normal global/crop 두 mild augmented view를 사용한다. Native CLIP normalize,224입력, random resized crop scale.9~1.0/ratio.95~1.05, brightness·contrast.1/saturation.05, flip/hue 변경 없음. Temporal clip이나 새로운 phase supervision은 추가하지 않는다.
- 목표는 frozen CLIP-L 원본 특징 cosine 보존 +.1×동일 공정·다른 영상 negative를 사용하는 양방향 contrastive loss(temperature.1)다. Frozen teacher는45 특징 cache이며 test pseudo-label이나 anomaly label을 사용하지 않는다. 서로 다른 정상 영상의 비슷한 phase가 false negative가 될 수 있다는 한계를 보고한다.
- Epoch당4096 train view, fixed validation2048 view, effective batch64를 scene/video/global-crop balanced sampling한다. 반복 sampling 수와 실제 고유 영상/표본 수를 구분한다.
- 최대60epoch(3840 optimizer steps), 최소10epoch,2epoch warmup 후60epoch horizon cosine schedule. Validation loss의 의미 있는 개선(min_delta1e-4)이8epoch 없으면 최소 학습 후 early stop한다. Epoch0을 포함한 최저 validation loss의 비붕괴 checkpoint를 선택한다. 마지막 epoch를 자동 선택하지 않는다. 최대 epoch에 닿으면 수렴을 단정하지 않고 curve/종료 이유를 보고한다.
- AdamW, weight decay.01, gradient clip1.0. 실용적인 각 적응 recipe로 L lr1e-4 /P lr1e-5 /D lr3e-6을 사용한다. 서로 다른 LR·early stopping 때문에 순수한 trainable scope만의 인과 비교나 동일 실제 step 수 비교로 해석하지 않는다. 모든 step·epoch·학습률·시간을 공개한다.
- Training/validation forward는 BF16 autocast, optimizer/master weight는 FP32이며 추론 특징 추출은45와 같은 FP32다. TF32 비활성. 동일 고정 augmentation/batch로 선택 checkpoint를 재로드해 validation을 재현한다.
- Validation feature mean std와 effective rank를 epoch0 대비 측정한다. 둘 중 하나가 초기의50% 미만이면 collapse 의심 checkpoint로 선택에서 제외한다. 이 간단한 gate가 모든 과적합을 막는다고 주장하지 않는다. Train/val gap, teacher drift, 선택 epoch, 최소 이후 하락을 함께 보고한다. Frozen parameter/buffer의 exact preservation과 trainable update를 확인한다.
- Epoch 경계의 checkpoint·optimizer·RNG·sampler 상태를 저장해 실패 시 동일 trajectory 재개가 가능하게 한다. 프로세스가 살아 있으면 재시작하지 않는다. Best checkpoint의 hash와 학습 protocol을 보존한다.

## 정상 모델과 개발 평가

모든 학습과 checkpoint 선택을 완료·동결한 뒤 F대조와L/P/D의9개학습run에 대한 정상111/test66 특징을 추출한다. Detector/box/tracks/phase/R01~R04 transition·dwell rule은45와 동일하다. Phase별 PCA95%/rank32/min10, 정상 CDF/q99를 다시 적합하며 정상 calibration video holdout을 수행한다. 기존 F prediction과 process score의 exact equality를 검사한다.

모든 예측을 고정한 후 strict31,550valid frames/66events로 AUROC/AP/FPR/recall/event coverage·공정별/영상별 결과를 계산한다. 본 개발 test로 epoch·LR·seed를 선택하지 않는다. 세 seed의 평균·표준편차는 기술 통계이며 독립 CI가 아니다. Checkpoint val loss가 좋아도 AD가 나쁘면 그대로 남긴다. 학습·추론 비용, 미탐/오탐, rank support/variance 변화와 normal holdout을 함께 보고한다.

## 완료와 다음 단계

실제9학습run/선택 checkpoint/복원검증/전체추론/독립 지표검증을 마쳐야46 완료다. 결과·의의·한계·추천순3개 및 README를 작성·검증해 GitHub에 업로드한다. 그 뒤46 결과로 다음 한 요소를 선택한다.45에서 확인한 PCA 용량, patch/object feature 수, 시간 관측 또는 detector 병목 등은 후보이며 이후 실험 전체를 미리 확정하지 않는다.

근거: [LoRA 원 논문](https://arxiv.org/abs/2106.09685), [공식 CLIP-L checkpoint](https://huggingface.co/openai/clip-vit-large-patch14). 기존 방법의 적용 비교이며 새로운 LoRA 알고리즘이라는 주장은 하지 않는다.
