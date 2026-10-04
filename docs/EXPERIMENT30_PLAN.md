# 다음 실험 계획 — 30

상태: 실험 29 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

선택된 anchor/target track 쌍이 바뀔 때 관계 평균을 초기화하면, 서로 다른 객체의 특징 혼합이 만드는 잠재 전이와 오탐을 줄일 수 있는가?

실험 29에서 정상 관측 3→2 6회 모두 현재 평균에 여러 track 쌍이 섞였다. 두 calibration 반례는 이전/현재 track이 같아도 더 이른 track의 특징이 평균에 남았고 raw descriptor의 최근접 중심은 2→2였다. 정상 유효 관계 중 FIT 114/1,123, calibration 26/297 samples에서 쌍 혼합이 확인됐다. 지속적인 바이스/제품 역할 혼동은 별도 문제이며 이번 변경으로 해결된다고 가정하지 않는다.

## 이번 변경과 대조

1. **관계 descriptor의 인과적 평균 경계만** 바꾼다. 같은 연속 관측 track 쌍에서는 기존 길이 3 평균을 유지한다. anchor 또는 target track ID가 바뀌면 이력을 비우고 현재 raw descriptor부터 시작한다. 누락 시 기존 이력 초기화·마지막 phase 유지 규칙은 그대로다. 객체 후보·선택 규칙·semantic margin·bbox·valid mask·현재 raw descriptor를 바꾸지 않는다.
2. 실험 18의 normal FIT phase 중심·location/scale·area gate를 고정한다. KMeans 재학습이나 ID 재정렬을 이번에 함께 하지 않는다. 변경된 평균을 기존 중심에 할당한다. 따라서 phase ID의 좌표계는 그대로지만 기존 중심의 학습 분포와 달라질 수 있다는 한계를 보고한다.
3. 전체 정상 FIT/calibration에 새 phase를 적용하고, 그 phase를 사용하는 appearance PCA·process 확률·dwell·정상 reference/q99는 정상 데이터로 다시 적합한다. 기존 점수 모델에 새 phase만 끼워 넣어 성능을 평가하지 않는다. 객체 임베딩과 pooled PCA는 입력이 같으므로 보존 여부를 확인한다.
4. 채택하지 않은 실험 28의 관측 CDF 대신 **실험 27의 full-transition CDF + consecutive-observed gate**를 두 경로의 공통 기준으로 사용한다. 외형 hold/pool/age 세 gate를 모두 유지하며 하나를 test 성능으로 선택하지 않는다. τ=56은 원래 normal FIT에서 동결한 값을 사용한다.
5. 평균 변경으로 phase별 FIT 표본과 자연 PCA rank가 달라질 수 있으므로, 정상 FIT만으로 다음 rank 대조를 구성한다. 기존 평균(control)과 reset 평균을 먼저 원래 variance .95/maxrank32 규칙으로 적합한다. 양쪽에 지원되는 같은 role×phase bank의 공통 rank는 두 FIT rank와 실험 27의 해당 rank 상한 중 최솟값으로 정한다. 한쪽에만 지원되는 bank는 그쪽의 자연 FIT rank를 유지하고 지원 차이를 별도 보고한다. 각 모델에 맞는 완전한 rank map을 정상 FIT에서 동결한다. pooled bank는 자르지 않는다.
6. 비교 구성은 `30_control_hold/pool/age`와 `30_reset_hold/pool/age`다. 공통 rank가 실험 27과 같지 않으면 control이 실험 27을 그대로 재현한다고 주장하지 않는다. 실험 27은 저장된 역사적 참조로만 함께 제시하고 주 결론은 동시 대조에서 낸다. phase별 표본 수·support와 PCA/공정 재적합 효과는 이번 전체 파이프라인 변경에 포함됨을 명시한다.

## 검증 순서

- 구현/입력/기존 모델/계획 hash를 정상 준비 전에 기록한다. track 교체·누락·동일 쌍 유지·영상 시작·인과적 prefix를 검사한다. reset 경로에서 유효 평균의 mixed-pair window가 0인지 검증한다. 동일 객체의 track 단절도 초기화될 수 있음을 보고한다.
- 정상 FIT 20개로 파생 phase와 rank map을 만들고 동결한다. 정상 calibration은 rank나 hyperparameter 선택에 쓰지 않는다. control이 기존 phase/선택/descriptor를 재현하고 reset이 선택/valid/raw geometry를 보존하는지 확인한다.
- 각 후보 정상 모델이 PCA·완결 dwell support와 calibration을 만들 수 있는지 검사한다. 6개 구성×5개 정상 영상 holdout을 수행한다. finite/[0,1]/q99<1 및 필요한 support를 통과한 후보만 테스트한다. 정상 실패를 임계값 조정이나 test 기반 수정으로 숨기지 않는다.
- 정상 audit를 마친 뒤 고정된 변환으로 R04 테스트 19개를 각 적격 후보에서 한 번 평가한다. 기존 raw feature/검출 결과만 재사용하며 VLM/검출기/CLIP을 다시 호출하지 않는다.
- AUROC/AP, 자체 정상 q99에서 FP/TP·구간/조건부 지연, 정상 holdout·phase support·관측 평균 혼합·process/dwell 가용성·공정 독자 기여를 보고한다. q99가 다르면 동일 reset 점수에 paired control q99를 적용한 사후 진단으로 임계값 효과를 분리한다.
- 정상 3→2 반례의 phase/전이 변화와 새로운 오류도 추적한다. 해당 두 반례를 제거하는 것만으로 전체 개선이라 판정하지 않는다.
- 결과·의의·보완점·추천순 3개를 보고하고 GitHub에 업로드한다. 다음 변경은 결과에 따라 결정한다.

## 한계

고정 phase 중심은 기존 평균에서 적합됐고 새로운 평균에 최적이라는 보장이 없다. track ID는 실제 identity 정답이 아니며 초기화가 노이즈나 불연속성을 늘릴 수 있다. 지속적인 역할 오류·가림·이상 객체 누락은 남는다. 반복 R04 개발, 작은 정상 calibration과 독립 녹화 그룹/semantic GT/FPS 부재의 한계를 유지한다. 성능 상승만으로 novelty를 주장하지 않는다.
