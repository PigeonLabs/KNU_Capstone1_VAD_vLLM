# 다음 실험 계획 — 26

상태: 실험 25 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

phase bank 지원 부족으로 실제로 pooled PCA를 사용하는 경우도 pooled CDF를 선택하게 하면, 같은 bank인데 요청 이름 때문에 보정 점수가 달라지는 현상을 제거할 수 있는가?

실험 25는 같은 요청 경로의 점수 불변성을 달성했다. 그러나 25_hold→25_pool에서 실제 bank는 같은데 요청이 달라진 특징 관측 531개 중 519개의 보정 점수가 달랐다. 25_hold→25_age에서는 504개 중 492개였다. 테스트 초기 미관측은 기존에도 pooled bank를 사용하지만 336프레임의 Combined score가 달랐다. 이 구간의 경보는 0개로 같았어도 ranking에는 영향을 줄 수 있다. 새 임계값을 이전 정상 q99로 되돌린 사후 진단에서는 경보 변화가 0개였으므로 이번에는 임계값 변경보다 CDF 선택 규칙에 집중한다.

## 이번에만 구체화하는 변경

1. 실험 25_hold / 25_pool / 25_age의 세 gate와 두 full-normal CDF 생성 규칙을 고정한다. 새 구성은 26_hold / 26_pool / 26_age다. 정상 FIT 표본·PCA bank·공통 rank·phase/mask·τ=56·process·sampling을 보존한다. 이전 버전과 각각 짝지어 비교하고 하나의 gate를 선택하지 않는다.
2. 변경은 **보정 CDF 선택 기준**이다. 요청 이름을 보는 대신 실제 선택된 subspace key를 확인한다. key의 phase가 0 이상이면 기존 phase 요청 기준 CDF, key가 pooled이면 기존 pooled 요청 기준 CDF를 사용한다. 따라서 명시적 pooled와 지원 부족 fallback이 같은 역할의 pooled PCA를 사용하면 같은 reference를 받는다.
3. 두 reference의 정상 모집단/순서/수치는 실험 25와 그대로 유지한다. 첫째 reference는 원본 phase 요청에 지원 부족 fallback을 포함해 전체 calibration을 계산한 값이다. 이를 실제 phase bank 표본만으로 적합한 조건부 분포라고 부르지 않는다. 둘째도 전체 calibration을 pooled PCA로 계산한 값이다. 실험 20처럼 실제 route subset으로 reference를 축소하거나 다시 적합하는 변경이 아니다.
4. 실제 pooled bank를 쓰는 관측은 같은 canonical pooled residual 배열을 사용한다. phase 요청의 지원 부족 fallback과 명시적 pooled 계산의 batch 차이로 생기는 부동소수점 tie 차이도 없앤다. 수학적 PCA residual은 보존돼야 하며 이전과는 1e−12 허용 오차로 검증한다. 변경되지 않은 지원 phase 경로의 점수는 이전과 정확히 같아야 한다.
5. 같은 역할/같은 실제 bank/같은 특징은 gate 또는 요청 이름이 달라도 보정 점수가 정확히 같아야 한다. 초기 미관측, 관측된 phase 0의 지원 부족, 짧은 누락의 지원 부족을 따로 보고한다. 실제 bank가 달라지는 구간에서는 점수가 달라질 수 있다.
6. 정상 CDF 배열은 동일하게 유지하지만 최종 q99는 각 새 구성의 정상 결합 점수로 다시 적합한다. 점수 일관성과 경보 일관성을 구분하고 q99 변화가 있으면 같은 저장 점수를 이전 정상 q99에도 적용하는 진단을 별도로 기록한다. 주 결과는 각 구성의 자체 q99다. 테스트 라벨로 임계값을 선택하지 않는다.
7. 코드·설정·입력·두 reference 정의를 정상 실행 전에 고정한다. 동일 정상 영상 leave-one-out 5 fold를 사용해 양쪽 reference와 process/q99에서 제외 영상 비누출을 검증한다. 전체 정상 보정의 finite/[0,1]/포화/q99<1 가능성을 확인하고 실패한 구성도 숨기거나 임계값을 수정하지 않는다.
8. 가용성 검사를 통과한 세 고정 구성을 각각 한 번 테스트한다. 전체 Visual/Combined AUROC/AP·FP/recall·구간/지연, 고정 누락 나이 subset, 같은 bank/다른 요청의 특징 점수 변화 수를 공개한다. 기존 대비 변동을 지원 부족 CDF 변경과 q99 변경으로 나누고 정상 holdout도 비교한다.
9. 모든 bank/reference/process 불변, 실제 bank dispatch, canonical residual, 제외 영상 비누출과 점수·원본 라벨·지표를 재구성한다. 정의상 지원 부족 case의 보정이 바뀌는 것이지 객체 의미 정답이나 신규 일반화가 확보되는 것은 아니다.

이 단계는 보정 규칙의 일관성을 강화하는 구현이다. 일관성 확보와 탐지 성능 개선은 별개이며 부정적인 결과도 그대로 보고한다. 반복 R04 개발, 역할/phase GT·독립 녹화 그룹·FPS 부재를 유지하고 후속 전체 실험을 미리 기획하지 않는다.
