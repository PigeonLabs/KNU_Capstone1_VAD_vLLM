# 다음 실험 계획 — 36

상태: 실험 35 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

현재 confirmed 객체 선택으로 얻은 정상 관계 descriptor에 phase 모델을 다시 적합하면, 과거 선택 정책에서 학습한 중심을 유지하는 경우보다 정상 phase 지원과 후속 모델의 정합성이 나아지는가?

실험 35는 track 왕복과 unknown-entry를 줄였으나 combined ranking·frame recall·normal holdout이 나빠졌다. 체류 지원은 5→1 맥락, test 체류 가용성은 2,121→429 frames, 공정 독자 경보는 0이 됐다. 현재 유효 FIT phase 0/1/2/3은 2/697/132/292 samples인데 중심/scale은 실험 18의 다른 선택 정책에서 만든 것이다. 이 불일치가 원인인지는 아직 가설이다. 이번에는 선택기 확인 횟수를 더 조정하지 않고 **phase 모델 적합만** 대조한다.

## 한 구성 요소 변경

대조군은 실험 35 confirmed의 객체 선택과 실험 18 고정 phase 모델이다. 변경군은 같은 객체 선택/관계 descriptor를 사용하면서 정상 FIT 20영상의 **location(median), scale(IQR 및 기존 floor .05), KMeans 중심**을 재적합한다.

- anchor/target vocabulary·검출/추적/CLIP 캐시·role margin·3회 causal median·면적 상한·2회 확인 선택·쌍 경계 평균 reset을 그대로 유지한다. **면적 gate를 다시 fit하지 않는다.** raw 객체 선택·valid mask·smoothed relation descriptor가 두 구성에서 정확히 같아야 한다.
- 정상 FIT의 valid descriptor만 적합한다. calibration/test descriptor는 center/scale·ID ordering 학습에 사용하지 않는다. 최소 정상 지원, K=4, seed=42, n_init=10, scale floor 등 기존 phase 학습 설정을 유지하고 K/seed 탐색은 하지 않는다.
- 새 cluster ID는 기존 학습 규약처럼 정상 FIT의 median relative frame position으로 정렬한다. 이는 cluster 명칭을 정하는 규칙이며 action GT나 test-time 입력이 아니다. control과 새 숫자 ID를 같은 의미 상태로 간주하지 않는다.
- 정상 FIT contingency에 Hungarian mapping을 적용한 permutation 보정 비교는 **진단용으로만** 동결한다. inference label이나 grammar를 사후 성능에 맞춰 재매핑하지 않는다. 원래 ID 교차표와 permutation 보정 일치율을 함께 보고한다.
- 새 phase용 process 기록은 같은 R04 정상 object vocabulary와 `latent_0..3`, self/next/cycle template를 유지하되 모델 출처를 명시한다. 이 template가 실제 공정 문법이라는 주장은 하지 않는다. 새 VLM·검출기·encoder 호출이나 로컬 서버 설정 변경은 없다.

## 정상 적합·보정

1. 코드/설정/계획·정상 입력·대조 모델을 동결한다. control은 실험 35 confirmed cache를 재현한다. 새 모델의 학습에 calibration/test가 섞이지 않았는지 확인한다.
2. selection/valid/descriptor 불변, phase 추론의 prefix·미래 비참조, 좌표/scale 유효성·4개 cluster 성립을 검증한다. normal FIT raw/scaled dispersion, cluster support·영상별 집중, 전이 coverage, strict complete/censored/unknown-entry를 비교한다. collapse/지원 실패는 실패로 보고하며 test를 보며 K나 gate를 바꾸지 않는다.
3. 외형 PCA·전이·기존 체류를 각 phase로 다시 적합한다. 공유 phase bank rank는 양 구성 자연 rank와 실험 35 상한의 최소값으로 통제한다. ID가 바뀌므로 rank 대조만으로 같은 의미 bank가 된다고 주장하지 않는다. 새/소실 bank와 표본/맥락 지원 변화는 별도 보고한다. pooled PCA는 동일해야 한다.
4. 정상 calibration CDF/q99와 5개 leave-one-video-out 보정을 수행한다. FIT는 고정하고 held-out 영상은 보정에서 제외한다. hold/pool/age와 τ=56을 유지하고 finite/bounded·support·q99<1 검증 결과로 테스트 가능성을 정한다.
5. 이전 24개 정상 경계의 선택은 동일해야 하며 이미지 재선정은 하지 않는다. 그 경계의 phase/거리/지원 변화는 metadata로 비교한다. 역할/action 정확도를 새로 추정하지 않는다. 학습 모델·정상 결과·진단 mapping·threshold를 동결한다.

## 테스트 및 판정

정상 checkpoint 뒤 기존 R04 test 19영상/8,154 frames를 평가한다. 세 경로 모두 AUROC/AP, 자신의 정상 q99에서 FPR/recall, 구간 탐지/조건부 지연, Visual 대비 전이·체류 독자 기여를 보고한다. 임계값이 다르면 동일 변경군 점수에 control q99를 적용한 대조는 사후 진단으로만 사용한다. test label로 centers/mapping/threshold를 고르지 않는다.

phase 지원 증가가 안정적 공정 관측 증가인지, 단순 cluster 재분할인지 구분한다. strict complete 부족과 역할 오류는 center 재학습으로 자동 해결되지 않는다. 반복 R04는 개발 자료이며 독립 일반화·의미 phase 정확도·통계적 유의성·novelty를 주장하지 않는다. 완료 후 결과·의의·보완점·추천순 3개를 보고/업로드하고 다음 변경 하나를 고른다.
