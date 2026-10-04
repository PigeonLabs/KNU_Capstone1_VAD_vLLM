# 다음 실험 계획 — 28

상태: 실험 27 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

공정 결합에서 사용하는 관측 쌍과 정상 전이 CDF의 모집단을 맞추면, 남은 전이 점수 간섭이 줄어드는가?

실험 27은 관측 쌍만 결합해 정상 holdout FP 40→32, 테스트 FP −16/TP −4를 얻었으나 공정 독자 추가는 여전히 정상 16/이상 4프레임이며 새 구간 탐지는 없다. 연속 관측 subset의 Combined AUROC 0.6358은 Visual 0.6560보다 낮다. 정상 상태별 전이 CDF는 여전히 전체 전이 477개에서 만들어졌으나 결합에 쓰는 관측 쌍은 247개다. 이전 상태 0은 전체 20개/관측 0개라 지원 부족을 함께 검사해야 한다.

## 이번에만 구체화하는 변경

1. 27_hold/pool/age 각각에 대응하는 28_hold/pool/age를 만든다. 외형 gate 승자를 선택하지 않는다. 특징·phase/mask·FIT/PCA·외형 CDF·실제 bank dispatch·τ=56·체류·sampling·max 결합을 고정한다.
2. **정상 calibration 전이 reference의 포함 기준만** 실험 27의 `i>0 AND valid[i-1] AND valid[i]`로 바꾼다. 이전 상태별 CDF와 지원 부족 시 사용하는 global process CDF 모두 같은 기준으로 만든다. 기존 reference의 raw transition 값 계산식과 정상 FIT 전이 확률/allowed grammar는 유지한다. 정상 FIT을 관측 쌍만으로 재학습하는 변경은 이번 범위가 아니다.
3. 이전 상태별 minimum support=10은 기존 값을 그대로 쓴다. 지원이 부족하면 전체 관측 쌍의 global reference로 fallback한다. global 관측 쌍도 10개 미만이면 보정 불가를 명시해 실패로 기록한다. cutoff를 낮추거나 누락 표본을 다시 섞지 않는다. 보정 함수의 기존 midrank와 strict `>`를 유지한다.
4. 추론의 consecutive-observed gate는 실험 27 그대로다. gate가 닫힌 부분은 최종 transition 기여 0이고 raw/보정/gated transition은 모두 보존한다. 첫 sample의 보정 출력이 달라져도 최종 기여는 0이다.
5. 각 구성의 최종 정상 Combined score로 자체 q99를 적합한다. 정상 reference 표본 감소로 q99=1/점수 포화가 생길 수 있다. clipping·임계값 변경·test 기반 수정으로 숨기지 않는다.

## 검증 순서와 판정

- 정상 평가 전 설정·코드·입력/hash·계획을 동결한다. 관측 쌍 포함/제외, 영상 경계, 상태별/global reference의 독립 재구성, 부족 지원/잘못된 mask 거부를 검사한다.
- 정상 FIT 20 / calibration 5를 유지하고 6개 구성×5개 normal holdout을 실행한다. 제외 영상은 모든 CDF/q99에서 빼고 FIT/PCA는 유지한다. reference별 표본·영상 수, fallback 사용, 점수 포화와 정상 경보를 보고한다.
- 각 새 후보의 전체 정상 및 holdout에서 finite/[0,1]/q99<1과 최소 global support를 만족할 때만 해당 후보의 R04 테스트 19개를 한 번 평가한다. 실패 후보는 정상 실패로 보고하고 임의 보정해 다시 테스트하지 않는다.
- raw transition·체류·외형 점수 및 FIT 모델 보존, 관측 gate 불변을 검증한다. 관측 쌍의 CDF 값이 바뀌므로 새 점수가 이전보다 작아야 한다는 제약은 두지 않는다.
- AUROC/AP, 정상 FPR/이상 recall, 탐지 구간/조건부 지연, 이전 상태·관측/재관측·지원/fallback subset, 공정 독자 TP/FP와 추가 탐지 구간을 보고한다. 임계값이 다르면 동일 새 점수를 이전 정상 q99에 적용해 효과를 분리하되 주 결과는 자체 q99다.
- 결과·의의·한계·추천순 3개를 보고하고 GitHub에 업로드한다. 이후 변경은 이번 결과에서 다시 선택한다.

## 한계

관측 쌍은 의미 정답이 아니며 재보정만으로 오검출을 교정할 수 없다. reference 수가 줄면 작은 표본/tie/지원 부족으로 오탐이 증가할 수 있다. 정상 FIT 확률과 calibration 모집단의 차이는 남아 있어 이번에는 CDF 효과만 해석한다. R04 반복 개발과 단일 분할, 독립 녹화 그룹·bbox/phase GT·FPS 부재를 유지한다. 성능 상승 자체를 novelty로 주장하지 않는다.
