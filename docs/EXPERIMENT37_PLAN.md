# 다음 실험 계획 — 37

상태: 실험 36 결과에서 고른 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

현재 관계 descriptor를 정상 median/IQR로 정규화한 뒤 좌표별 `asinh`로 극단값의 영향을 완화하면, 소수의 큰 좌표에 치우친 phase 분할이 달라지는가? cluster 지원과 같은 객체 쌍의 공정 전환 근거가 함께 개선되는지 확인한다.

실험 36의 유효 FIT phase 지원은 989/2/128/4이고, 소수 두 cluster의 6 samples(0.53%)가 median 중심 제곱 에너지 44.09%를 차지했다. 상대 x와 anchor y의 에너지 비중은 42.89%/42.05%, 상대 x 최대 절댓값은 96.76이다. 같은 쌍의 정상 phase 전환과 strict complete는 0이다. 이 수치는 거리 편중 가설의 근거이며 KMeans inertia의 인과 분해가 아니다.

## 변경 하나: phase 거리 좌표 변환

- control은 실험 36 refit의 선형 정규화 `z=(x−median)/scale` 및 중심이다. 변경군은 **같은 median/scale로 `u=asinh(z)`**를 만든 뒤 정상 FIT 20영상에서 KMeans 중심을 재학습한다. 추론에도 동일한 변환을 적용한다. `asinh`는 부호를 보존하는 단조·비선형 압축이며 유계 함수가 아니다. 계수나 clipping threshold를 추가하지 않고 다른 변환을 탐색하지 않는다.
- 면적 gate·anchor/target vocabulary·검출/track/CLIP·semantic/causal median·2회 확인 선택·valid mask·쌍 경계 평균 reset·6차원 raw/smoothed descriptor는 완전히 고정한다. median/scale도 실험 36 정상 FIT 값과 정확히 같아야 한다. bbox를 삭제하거나 작은 box를 오검출로 재라벨링하지 않는다.
- K=4, seed=42, n_init=10, 최소 지원 및 scale floor=.05를 유지한다. FIT median relative position으로 ID를 정렬한다. 정상 FIT contingency의 Hungarian 대응표는 진단용으로만 동결하고, inference/grammar를 성능에 맞춰 매핑하지 않는다.
- 같은 정상 vocabulary, latent_0..3과 self/next/cycle template를 유지한다. 새 VLM·검출기·encoder 호출이나 로컬 서버 설정 변경은 없다. 기존 체류 및 같은 쌍 전이 결합을 유지해 거리 변경과 체류 gate 변경을 섞지 않는다.
- 정상 phase별 PCA·전이·체류·CDF/q99는 각각 재적합한다. 공통 numeric bank rank는 양 구성 자연 rank와 실험 36 상한의 최소값으로 통제하며, 신규/소실 bank·표본·rank 변화를 보고한다. ID의 의미 동일성은 가정하지 않고 pooled PCA 동일성을 검사한다. control rank가 달라지면 실험 36의 정확한 재현과 구분한다.

## 정상 검증과 동결

1. 코드/계획·정상 입력·기존 모델을 먼저 동결한다. 정상 단계에서 test/label 접근을 막고 phase 적합에 FIT 20영상만 쓰였는지 기록한다.
2. 입력 불변, descriptor/선택/valid 동일성, `asinh` 수치 계산, 새로운 모델의 저장/복원, prefix·미래 비참조를 검증한다. 별도 계산으로 적합·ID 정렬·추론을 재구성한다.
3. cluster별 표본/영상 집중·정규화 에너지·새 거리의 극단값 영향·대응표·같은 쌍 전환·strict complete/censored/unknown-entry를 비교한다. 다른 거리 단위의 inertia 감소를 직접 정확도 상승으로 해석하지 않는다. 균등한 cluster 자체가 목표도 아니다.
4. 정상 calibration과 5개 leave-one-video-out의 CDF/q99를 검사한다. FIT는 고정하고 held-out 영상은 보정에서 제외한다. 세 fallback과 τ=56을 유지한다. finite/bounded·기존 support·q99<1을 만족한 구성만 test로 진행한다. 모든 구성이 실패해도 K/threshold를 바꾸지 않고 실패를 보고한다.
5. 이전 24개 정상 사례의 선택은 같아야 하며 phase/거리/bank 지원 metadata만 비교한다. 이미지 재선정이나 action 정확도 추정을 하지 않는다. 초기 phase0 bank의 의미/지원 변화와 첫 관측 전 fallback도 별도 기록한다. 학습 모델·정상 진단·보정을 동결한다.

## 테스트와 판정

정상 checkpoint 이후 기존 R04 test 19영상/8,154 frames를 평가한다. AUROC/AP, 각 정상 q99의 FPR/recall, 구간 gained/lost·조건부 지연, Visual 대비 공정 독자 경보, 체류 지원과 같은 쌍 관측 근거를 보고한다. 임계값이 다르면 같은 변경군 점수에 control q99를 적용한 사후 대조를 추가한다. test label로 변환/중심/threshold를 선택하지 않는다.

극단값 압축이 실제 공정 변화까지 약화시킬 수 있다. 새로운 phase가 일부 성능을 높여도 역할 오류나 실제 동작 문법을 해결했다고 부르지 않는다. 같은 R04의 반복 개발이므로 일반화·통계적 유의성·novelty는 입증하지 못한다. `asinh` 자체는 표준 변환이다. 완료 후 결과·의의·보완점·추천순 3개를 정리해 업로드하고, 그 결과에서 다음 변경 하나를 고른다.
