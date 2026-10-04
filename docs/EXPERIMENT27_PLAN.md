# 다음 실험 계획 — 27

상태: 실험 26 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

이전·현재 관계가 모두 관측된 전이만 최종 공정 점수에 포함하면, 유지되거나 초기화된 phase의 전이가 만드는 간섭을 줄일 수 있는가?

실험 26은 실제 bank의 보정 불일치를 제거했지만 Combined AUROC 0.6650–0.6669는 Visual 0.6875–0.6899보다 낮았다. process 배열은 그대로였으며 현재 transition은 관측 여부와 무관하게 held phase를 사용한다. 체류 점수는 이미 관측 진입과 정상 지원을 요구하지만 유효 범위는 5.11%다. 따라서 이번에는 전이의 결합 근거를 분리한다. 전체 공정 신호 제거 또는 가중치 탐색을 동시에 하지 않는다.

## 이번에만 구체화하는 변경

1. 26_hold/pool/age를 각각 기준으로 27_hold/pool/age를 만든다. 외형 gate의 승자를 고르지 않고 세 쌍 모두 보고한다. 특징·phase/mask·FIT/PCA·두 외형 reference·실제 bank dispatch·τ=56·process 확률/정상 reference·체류 모델은 보존한다.
2. sampled frame i의 transition 기여 조건을 `i>0 AND relation_valid[i-1] AND relation_valid[i]`로 고정한다. 같은 phase 유지 전이도 이 조건을 만족하면 포함한다. 첫 sample, 누락, 누락 직후 첫 재관측의 transition 기여는 0으로 둔다. test 길이·미래 관측을 참조하지 않는다.
3. raw transition 및 기존 보정 transition은 별도 진단 필드로 보존한다. gate는 보정 이후 결합에만 적용한다. `process=max(gated_transition, valid_dwell_score)`, `combined=max(visual,process)`다. 체류의 기존 valid 규칙과 점수는 바꾸지 않는다. 전이 확률을 관측 쌍만으로 다시 적합하거나 CDF를 재적합하는 변경은 이번 범위가 아니다.
4. 정상 최종 결합 분포로 자체 q99를 다시 적합한다. q99가 바뀌면 동일 새 점수를 기존 정상 q99에 적용하는 사후 진단으로 score 효과와 threshold 효과를 구분한다. 테스트 라벨로 q99·가중치·gate를 선택하지 않는다.

## 검증 순서

- 입력/hash·설정·계획·구현을 정상 평가 전에 기록한다. gate mask의 인과성, 처음/누락/재관측 경계, 기존 경로 불변, 잘못된 mask 거부를 검증한다.
- 정상 FIT 20개와 calibration 5개를 유지한다. 각 제외 영상이 외형·process reference와 q99에서 빠지는 6개 구성×5개 normal holdout을 수행한다. gate가 닫힌/열린 subset과 정상 이전 상태 support도 진단하되 support cutoff를 추가 선택하지 않는다.
- 모든 후보가 finite/[0,1]/q99<1 등 보정 가능성을 만족하면 세 새 구성 모두 R04 테스트 19개를 한 번씩 평가한다. 실패 후보가 있으면 라벨 기반 수정 없이 원인과 실패를 보고한다.
- global/crop 외형 점수·PCA·두 CDF·원래 transition/dwell이 같은지 독립 재구성한다. gate 적용 process는 기존보다 클 수 없고, 같은 임계값에서는 추가 경보가 없어야 한다. 자체 q99 비교에서는 추가 경보가 가능하므로 분리한다.
- AUROC/AP, 정상 FPR/이상 recall, 구간 탐지·조건부 지연, 관측 쌍/누락/재관측 및 기존 age subset을 보고한다. 같은 자체 q99에서 visual-only와 fused 경보의 독자 FP/TP, transition/dwell 지배 범위를 분해한다. Visual-only의 별도 최적 임계값 실험으로 해석하지 않는다.
- 표/필요한 그림, 의의·실패·추천순 개선 3개를 보고하고 단계별 GitHub에 업로드한다. 다음 변경은 이 결과에서 다시 선택한다.

## 판정의 한계

관측 mask는 semantic 정확도 정답이 아니다. 누락 자체가 이상 단서일 수 있으므로 gate로 실제 탐지를 잃을 수 있다. 기존 전이 확률/CDF를 보존하는 실험이므로 관측 전이만의 확률 모델은 검증하지 않는다. R04 반복 개발·정상 calibration 5개·독립 녹화 그룹/FPS/bbox·phase GT 부재를 유지한다. 성능 상승 자체를 novelty로 간주하지 않는다.
