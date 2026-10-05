# 다음 실험44 계획 — 독립 주석으로 현재 학습 파이프라인 검증

상태: **계획이며 실행 결과 없음**. 40~43에서 현재 데이터의 공유/공정별 학습 모듈 구현을 마쳤다. 43의 추천순1을 다음 검증 단계로 선택한다. 다른 개선 후보까지 동시에 도입하지 않는다.

현재의 높은 phase agreement는 weak geometric teacher와의 일치율이다. R02 텍스트 teacher의 의미 오류, 희소한 상태 검증 자료, 반복 사용한 개발 장면 때문에 새로운 head 학습을 더 반복하는 것만으로 논문 근거를 강화하기 어렵다. 다음은 기존 모델을 고정한 독립 검증을 먼저 수행한다.

## 필요한 자료와 분할

- R01~R04에서 지금 사용한 영상과 다른 녹화 세션/물리 객체/작업 조건의 정상 및 이상 영상. 출처·촬영 세션·프레임 수·원래 시간 정보를 보존하고 FPS를 추정해 채우지 않는다.
- 사람 검수로 phase의 의미와 경계, 복수 가능한/가려진/정의 불가 구간, 객체 identity 및 split/merge 정책을 명시한다. 기존 latent cluster ID를 그대로 정답으로 복사하지 않는다.
- 모델 수정용 자료와 최종 평가 그룹을 녹화 세션 단위로 분리한다. 최종 그룹을 보기 전에 평가 범위·metric·unknown 처리 규칙을 고정한다. 새로운 입력/주석이 없으면 이번 test를 독립 test라고 이름만 바꾸지 않는다.

## 고정 대조와 판정

기존41,43 teacher,43 linear,43 adapter와40의 visual A/B/C/D를 결과에 따라 고르지 않고 사전 지정된 대조로 유지한다. 이번 단계에서 detector·representation·phase head를 재학습하거나 test 기반 threshold를 조정하지 않는다. 현재 calibration threshold의 전이 가능성과 정상 오탐률 변화부터 확인한다.

사람 phase GT에 대한 class별 precision/recall·경계 오차·unknown coverage, identity IDF1/HOTA 및 false-link, AD AUROC/AP·FPR/recall·event coverage를 별도로 보고한다. Weak-target agreement는 참고 진단으로만 남긴다. 데이터 부족으로 측정하지 못하는 항목은 미측정으로 표기한다.

자료가 준비되면 이 단계를 실행하고 결과·의의·한계·추천순3개를 갱신해 업로드한다. 현재는 독립 자료가 제공되지 않았으므로 결과나 완료 지표를 생성하지 않았다. 이는 다음 검증 계획이며, 완료된40~43의 학습 결과와 구분한다.
