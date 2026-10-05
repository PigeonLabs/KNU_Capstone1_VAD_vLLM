# 실험41 계획 — 정상 약지도 기반 공유 GroundingDINO 학습

상태: **계획만 수립, 미실행**. 실험40 결과·검증을 완료한 뒤 선택한 다음 변경이다. 이번 문서는41 하나만 구체화하며42 이후 실험을 확정하지 않는다. [실험40 보고서](EXPERIMENT40.md)의 추천순1을 따른다.

## 선택 근거와 질문

40-C는 frozen MobileCLIP2보다 공정 평균 Combined AUROC0.6584→0.6702로 개선됐지만 평균 recall은17.01%, R02는5.02%, R04는11.42%에 머물렀다. Detector/box/role/track/phase를 고정했기 때문에 공유 visual representation 학습이 관측 오류를 해결한 실험은 아니다. 이전 실험33의 역할 혼동·객체 선택 문제도 남아 있다. 이번 수치만으로 detector가 낮은 recall의 원인이라고 단정하지 않고 **공유 detector의 정상 도메인 적응이 입력 관측과 최종 경보를 함께 개선하는지** 검증한다.

40-D는 정상 validation loss가 더 낮아도 anomaly ranking이 더 낮았다. 이를 반영해41에서도 normal 학습 loss 감소만으로 개선을 선언하지 않고 frozen detector 대조와 공정별 최종 평가를 유지한다.

## 변경 하나와 대조

- 41-Frozen: 현재 GroundingDINO-tiny 가중치를 그대로 사용한다.
- 41-Learned: 같은 초기 가중치에서 **공유 decoder와 bbox prediction heads**를 정상 자료로 학습한다. 첫 단계에서는 vision backbone, text backbone, fusion encoder를 고정한다. 모델을 복제한 뒤 실제 trainable 목록·gradient·parameter 변경·동결부 불변을 smoke에서 확인한다. 무작위 초기화나 weight loading 실패를 학습 결과로 취급하지 않는다.
- Visual encoder는40의 **A/B/C_s42/D_s42/C_s43/D_s43/C_s44/D_s44 모두 고정해 재사용**한다. 40 테스트에서 가장 좋은 encoder나 seed를 골라 대조군을 없애지 않는다. Detector 한 요소의 효과를 각 표현 안에서 paired 비교한다.
- Vocabulary/prompt, tracker 알고리즘, process별 phase 추정 알고리즘과 기존 phase 파라미터는 유지한다. 새 detector가 만드는 boxes/tracks/phase 관측의 변화는 전파 효과로 기록한다. 새 관측에 대한 process별 appearance/transition/dwell/CDF/q99는 같은 정상 분할·규칙으로 다시 적합한다. 구조나 hyperparameter를 바꾸지 않는다. Detector가 바뀌어 생기는 downstream 재적합을 추가 신경망 학습으로 혼동하지 않는다.
- R02 phase 및 R04 anchor의 기존 frozen CLIP auxiliary 경로도 유지한다. learned ReID·learned phase head·encoder 재학습은41에 함께 넣지 않는다. Transition/dwell은 통계 모델로 남긴다.

## 정상 학습 데이터와 약지도의 범위

40의 표현 train70영상/val19영상/calibration22영상을 그대로 사용한다. Detector 최적화는 train70만, checkpoint 선택은 val19만 사용한다. Calibration22와 test는 detector 학습·pseudo-label 생성·epoch 선택에서 제외한다.

먼저 공정/role별 정상 후보 annotation manifest를 만든다. 초기 후보는 frozen teacher의 confidence≥0.5, 같은 role의 최소3개 연속 sampled 관측과 bbox 연속성이 있는 검출을 사용한다. 공정별 최대256 training frame,64 validation frame을 영상·role·관측 상태가 한쪽으로 몰리지 않도록 선정한다. 실제 지원 부족은 수를 줄여 공개하며 낮은 confidence 후보를 몰래 추가하지 않는다. 같은 영상의 인접 프레임이 train/val로 나뉘지 않도록 영상 단위 분리를 먼저 적용한다.

별도 bbox 정답은 현재 없다. 자동 후보는 **pseudo-label**로 표시하며 검수 시 원본 영상, 역할 정의, box, 가려짐, 중복/누락, 검수 주체·근거·불확실성을 기록한다. 모델 보조 검수를 사람 정답이라고 부르지 않는다. 관심 객체가 누락됐거나 역할이 모호한 frame은 완전한 background 정답으로 학습하지 않고 학습 목록에서 제외한다. 확정 가능한 정상 후보가 충분한지 먼저 정상-only 데이터 감사로 확인한다.

Teacher와의 일치도를 detector accuracy/mAP/recall이라고 보고하지 않는다. 추후 독립 bbox 주석이 확보되면 그 범위의 보조 평가로 분리한다. 이번 학습 자체에는 anomaly label이 필요하지 않다.

## 초기 학습안과 동결 시점

초기안은 seed42,5epochs, batch2, AdamW lr1e-5/weight_decay1e-4, gradient clipping1.0, mild color/scale augmentation(좌우 반전 없음)이다. 현재 설치된 Transformers4.57.6 GroundingDINO는 `labels`의 class_labels/boxes를 받아 matching loss를 계산하는 경로가 있다. 실제 caption token↔role mapping, normalized cxcywh, padding/mask, trainable scope와 loss를 정상 smoke로 먼저 확인한다.

GPU/수치 smoke 후 필요한 실행 자원 조정과 정확한 입력 수·학습 파라미터를 기록하고 protocol을 동결한다. 그 뒤 test 결과에 맞춰 설정을 바꾸지 않는다. Epoch0도 후보에 넣어 고정 normal-val loss로 checkpoint를 선택한다. Normal-val pseudo-label loss는 teacher 일치성에 가까운 surrogate이며 최종 탐지 개선의 증거가 아니다. Epoch0이 선택되거나 adaptation이 유효하지 않으면 실패/무변화 결과를 그대로 보고한다. 단일 detector seed pilot이라는 한계를 남긴다.

## 평가와 판정

1. 정상-only annotation 지원·역할 혼동·지원 부족, 실제 gradient/update/복원 및 동결부 보존을 확인한다.
2. 역할별 관측 빈도, 누락 길이, box 변동, track 단절, phase 분포·dwell 지원을 정상 holdout에서 비교한다. 관측 빈도 증가가 올바른 검출 증가인지 검수 근거 없이 단정하지 않는다.
3. 각 detector×고정 representation×공정의 정상 모델/normal-video holdout/q99를 동결한다. Detector 변경으로 생긴 실제 PCA rank·normal support·임계값 변화도 공개한다.
4. 점수를 먼저 고정한 뒤40과 같은 strict label policy로 test를 평가한다. Visual/Combined AUROC/AP, 개별 q99 및 대조 q99 보조 비교, FPR/recall/구간 탐지를 공정별로 보고한다. 이전부터 관찰한 test는 개발용으로 명시한다.
5. Pseudo-label 복제만 하고 downstream 개선이 없거나 오탐·미탐 상충이 커지면 성공으로 포장하지 않는다. 결과·의의·보완점·추천순3개 → GitHub 업로드 → 다음 변경 선택 순서를 반복한다.

이 계획이 detector 학습의 성능 향상을 약속하지는 않는다. 실험40의 LoRA 결과도41의 최종 encoder 채택 결정으로 사용하지 않고 모든 기존 arm을 고정 대조로 유지한다.
