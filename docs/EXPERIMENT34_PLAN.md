# 다음 실험 계획 — 34

상태: 실험 33 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

현재 eligible anchor 중 semantic 근거가 가장 높은 후보를 먼저 선택하면, 이전 track을 무조건 유지하는 데서 생기는 역할 혼동을 줄일 수 있는가? 그 과정에서 track 단절·체류 지원 부족과 정상 오탐이 더 커지는가?

실험 33에서 선택된 anchor보다 temporal margin이 높은 eligible 대안이 FIT 84/calibration 28 samples에 있었고, 그중 82/28은 기존 track을 유지한 선택이었다. 01_0024·02_0176에서는 blade로 보이는 대안도 gate를 통과했지만 바이스가 유지됐다. 이 metadata 수치는 대안의 정답률이 아니다. 단일 선택 규칙 변경의 역할·연속성·점수 효과를 검증한다.

## 하나의 알고리즘 변경

대조군은 실험 31과 같은 prior-track 우선 선택이다. 변경군은 **anchor의 eligible 후보 중 temporal margin 최대를 우선**한다. margin이 정확히 같으면 이전 anchor track을 우선하고, 그다음 detector confidence, 마지막에는 캐시 index가 작은 후보를 선택한다. 별도의 차이 threshold, hysteresis, 가중치, margin 보정은 도입하지 않는다. 이전 track은 역할별로 선택이 있었을 때 갱신하고 missing에서 유지하는 기존 규칙을 따른다.

- raw 후보·role·confidence·CLIP 특징·track ID·면적 상한·margin>0 gate·3-observation causal median은 그대로다. target은 기존 선택 규칙을 유지한다. **valid mask와 target 선택이 두 구성에서 같아야 한다.**
- 관계 descriptor 평균은 쌍 교체 시 reset하는 실험 30 규칙을 사용한다. 실험 18 정상 중심/scale은 고정해 선택 변경의 효과를 분리한다. 평균·phase는 새 선택으로 재계산하되 중심을 재적합하지 않는다.
- 전이 결합은 실험 31의 같은 선택 쌍 연속 관측 gate, missing 외형 fallback은 hold/pool/age의 기존 정의를 사용한다. age τ=56, normal support 기준, 체류 점수 정의를 바꾸지 않는다. 실험 32 episode는 정확한 관측 근거를 진단하는 용도로 다시 추출하며 새 검열 분포를 도입하지 않는다.
- VLM/검출기/encoder 재호출과 서버 설정 변경은 없다. 기존 로컬 VLM 설정을 유지한다.

## 정상 단계와 동결

1. 구현·설정·계획·정상 입력을 hash로 동결한다. 동점/빈 후보/누락/target 불변/미래 비참조와 control 재현을 검증한다. 정상 FIT 20영상/calibration 5영상의 기존 분할을 유지한다.
2. 정상 특징을 두 선택 규칙으로 재구성하고 선택 index·쌍 변경·phase 변경·complete/censored/unknown-entry를 비교한다. 현재 규칙은 기존 실험 30 reset 캐시를 정확히 재현해야 한다.
3. 실험 33의 24개 고정 사례를 같은 원본으로 비교해 선택된 bbox의 역할 관찰과 불확실성을 모두 기록한다. 유리한 사례만 다시 고르거나 정확도를 계산하지 않는다. 정상 추가 사례가 필요하면 별도 metadata 규칙을 이미지 검토 전에 고정한다.
4. 정상 FIT만으로 두 구성의 외형 PCA를 재적합한다. 공유 bank rank는 각 구성의 자연 rank와 기존 실험 31 rank 상한의 최소값으로 맞춘다. 새 bank/지원 소실·표본 수 변화는 별도 보고하고 공통 bank의 순수 선택 효과라고 과장하지 않는다. control rank가 달라지면 재적합 control과 실험 31의 차이를 명시한다.
5. 정상 FIT 전이/체류 및 calibration CDF/q99를 각 구성에서 재계산한다. hold/pool/age 모두 보고하며 결과가 좋은 경로만 선택하지 않는다. calibration 영상 5개 leave-one-video-out 보정 검증에서 FIT는 고정하고 각 held-out 영상이 보정 자료에서 제외되도록 한다.
6. control 캐시, 정상 모델/점수의 재현 여부와 변경된 rank에 따른 예외를 기록한다. 정상 출력의 finite/bounded 여부·support·q99<1 경보 가능성을 확인하고 사전 test checkpoint를 만든다. 실패한 구성은 실패로 보고하고 test 라벨로 수선하지 않는다.

## 테스트와 결과 판정

정상 단계·설정·threshold를 동결한 뒤 R04 기존 test 19영상/8,154 frames를 평가한다. 이 장면은 반복 관찰한 개발 자료이며 독립적 일반화 검증이 아니다. AUROC/AP, 정상 q99에서 FPR/recall, 구간 탐지/조건부 지연, 외형 대비 공정의 독자 FP/TP를 모두 보고한다. q99가 달라지면 공통 control q99 대조를 진단용으로 추가하되 모델 선택에는 사용하지 않는다.

핵심 판정은 높은 margin 선택 구현 여부, valid/target 불변, anchor 역할 관찰, 선택 교체/episode 지원·phase 변화, 정상 holdout 및 최종 경보의 상충이다. 의미 정확도 GT가 없는 상태에서 margin 상승을 역할 정확도 상승으로 주장하지 않는다. 성능이 좋아져도 이 선택 규칙만의 novelty나 논문 주제를 확정하지 않는다.

## 예상 한계

margin 최대 선택은 오검출에 높은 CLIP 점수가 붙으면 잘못된 후보를 고르고, 작은 score 변동으로 잦은 track 교체를 만들 수 있다. 면적 gate에서 이미 제외된 blade는 선택할 수 없으며 target 혼동도 남는다. 고정 관계 중심은 새 descriptor 분포에 부적합할 수 있지만 이번에는 별도의 중심 재학습과 혼합하지 않는다. R04 정상 25영상, 단일 split/seed, 독립 녹화 그룹/객체·action GT/FPS 부재를 유지한다.

완료 후 결과·의의·보완점·추천순 3개를 갱신하고 GitHub에 업로드한 뒤 다음 변경 하나를 선택한다.
