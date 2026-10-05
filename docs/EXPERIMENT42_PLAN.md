# 실험42 계획 — 공유 learned association adapter의 제한적 검증

상태: **계획, 아직 학습·평가하지 않음**. 실험41 완료 결과의 추천순1을 구체화한다. 이후 실험 전체나 논문 주제는 정하지 않는다.

## 선택 근거

41에서 검수 사례의 객체 역할은 교정됐지만 C 평균 Combined AUC는0.6702→0.6421, recall17.01→6.93%로 낮아졌다. R04의 phase3 편중과 dwell 지원 부족도 확인됐다. ReID가 이 퇴행을 해결한다고 주장하지 않는다. 사용자가 지정한 GroundingDINO → learned ReID → learned phase head 순서에 따라, 다음은 **공유 추적 표현의 학습 가능성과 연결 효과**를 분리 검증한다. 공정별 phase 모델의 수정은 함께 넣지 않는다.

정상 R04 material/blade의 track ID 합계는223/268→247/286이며 관측 누락도 남았다. 이 수치는 identity 정확도가 아니다. 기존 IoU track ID를 정답으로 학습하면 오류를 증폭할 수 있다. 고 confidence 동일 역할의 동시 비중첩 후보는 R01~R03 train에0개, R04에163개뿐이므로 감독 지원 감사를 먼저 수행한다.

## 대조와 고정 범위

- Control: 실험41의 learned detector + 기존 class-gated Hungarian IoU tracker를 정확히 재현한다. 실험40 frozen detector 결과도 역사적 비교로 보존한다. 41을 성능상 우월한 모델로 채택했다는 뜻이 아니다.
- Frozen association: 같은 CLIP auxiliary crop feature의 cosine similarity를 쓰는 보수적인 재연결 경로다. Matching 정책 변경의 효과를 학습 효과와 분리한다.
- Learned association: 동일 matching 정책에 정상 pair로 학습한 작은 공유 residual metric adapter를 넣는다. GroundingDINO와 CLIP/MobileCLIP 가중치는 동결한다. 작은 head에 전체 foundation model FT를 적용하지 않는다.
- 실험40에서 가져온8개 visual arm을 유지한다. Association head는 공통 auxiliary feature로 한 번 학습하고, anomaly appearance encoder를 동시에 학습하지 않는다. 기존 phase 파라미터는 고정하고 track 변화가 phase 관측·statistical refit으로 전파되는 효과를 기록한다.
- Appearance/transition/dwell/CDF/q99는 공정별 같은 정상 분할·규칙으로 재적합한다. 41의 명시적 unavailable-dwell 정책과 최소 지원10도 유지한다.

## 정상 데이터 감사와 초기 학습안

Train70/val19/calibration22 영상 분할을 유지한다. Pair 후보는 정상 train/val에서만 만든다. 인접 프레임의 같은 역할·고 confidence·서로 유일한 높은 IoU 관측을 positive 후보로, 같은 프레임에서 분명히 분리된 같은 역할 객체를 negative 후보로 감사한다. 시점이 다른 동일 물체, 동일 물체의 split box, 잘못 검출된 배경을 근거 없이 negative로 쓰지 않는다. Raw frame/crop·box·시간·선택 근거와 검수 주체를 기록하며 모델 보조 검수를 사람 identity GT라고 부르지 않는다.

초기 adapter는512차원 정규화 CLIP feature 위의 rank8 residual projection(초기 identity), seed42, FP32,20epochs, AdamW1e-3/weight_decay1e-4, batch256이다. Positive cosine consistency와 검수한 same-role negative separation을 균형 있게 사용하고 작은 teacher-preservation 항으로 collapse를 억제한다. 정확한 pair 수·loss 가중치·batch 균형은 정상 지원 감사와 smoke 이후 **학습 전에** 동결한다. Epoch0을 포함한 정상 val objective로 checkpoint를 선택한다. 이 loss가 identity 정확도라는 주장은 하지 않는다.

기존 IoU 유효 매칭을 우선하고, 남은 후보에만 appearance 재연결을 허용하는 초기안을 검증한다. Role gate, max age2는 유지하며 대규모 순간 이동이나 모호한 매칭을 차단한다. Similarity gate는 정상 calibration의 검수 가능한 negative 근거로 정하고, 지원이 부족한 역할에는 appearance 재연결을 사용 불가로 표시하여 IoU로 돌아간다. 모든 역할에 학습된 ReID가 활성화됐다고 과장하지 않는다. 박스가 아예 누락된 프레임을 tracker가 복원한 검출로 생성하지 않는다.

## 판정과 다음 단계

1. Normal pair 지원·오류 유형·실제 gradient/update·동결 backbone·checkpoint 복원을 확인한다. 지원이 부족하면 threshold를 낮추거나 false identity 정답을 만들지 않고 그 범위를 제한한다.
2. Control의41 cache/정상 model·score 재현, association 변경 수·오연결 검수·누락 후 연결·동일 pair 길이·phase 변화와 dwell 가용성을 기록한다. 독립 identity GT 없이는 IDF1/HOTA를 보고하지 않는다.
3. 모델·normal calibration·모든 test score를 고정한 뒤 동일31,550 valid frames·66events 기준으로 평가한다. 새 q99와 paired 대조, AUROC/AP·FPR·recall·event coverage를 함께 보고한다.
4. 개선이 없거나 오연결이 늘면 실패/무변화 결과로 보고한다. 결과·의의·보완점·추천순3개를 정리해 GitHub에 업로드한 뒤, 그 결과를 바탕으로 다음 한 단계만 설계한다.

42는 ReID 효과의 파일럿이며 독립 녹화 그룹 일반화나 새로운 metric-learning 알고리즘의 제안이 아니다. 기존 test를 반복 관찰한 개발 실험이라는 범위를 유지한다.
