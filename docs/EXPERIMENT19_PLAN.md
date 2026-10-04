# 다음 실험 계획 — 19

상태: 실험 18 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

현재 객체 관계를 관측하지 못한 프레임에 초기/이전 phase를 확정적으로 적용하는 대신, 전체 정상 외형 모델을 사용하면 phase 조건화의 오용을 줄일 수 있는가?

실험 18은 gate 변동을 줄였지만 Combined AUROC 0.6655, 정상 오탐 379프레임으로 실험 17보다 악화됐다. 정상 FIT 1,960 samples 중 837개가 관계 미관측이고, latent 0의 126개 할당 중 직접 관측은 3개다. 테스트 관계 미관측은 2,555프레임이다. 이 사실이 성능 저하 원인이라는 결론은 아직 없으며, 외형 조건화 한 요소만 바꿔 검증한다.

## 고정할 변경

1. 실험 18의 모든 bbox·crop/global 특징·track·text embedding·집계 margin·관계 phase와 `relation_valid`를 재사용한다. 새 검출, VLM 호출, prompt/threshold/window/군집 재탐색은 하지 않는다. R04 분할, seed 42, PCA 분산 95%/최대 rank 32/최소 10개를 유지한다.
2. appearance의 **역할×phase PCA**에는 해당 sample의 `relation_valid=True`인 정상 FIT 특징만 넣는다. global-frame 역할에도 같은 관측 조건을 적용한다. 역할별 pooled PCA는 기존대로 모든 정상 FIT 특징으로 한 번씩 적합한다. 미관측 특징이 pooled bank에 중복 삽입되지 않게 한다.
3. normal calibration과 테스트에서 관계가 미관측이면 해당 역할의 pooled PCA를 사용한다. 관측된 phase여도 정상 지원이 최소 10개 미만이면 기존 pooled fallback을 유지한다. 해당 역할 pooled가 없으면 기존 global pooled fallback을 따른다. 미관측을 점수 0/이상으로 고정하거나 평가에서 제외하지 않는다.
4. 관계 phase 자체와 전이·체류 입력/학습/점수는 실험 18과 동일하게 유지한다. phase별 외형 bank 선택만 변경하고 역할별 정상 calibration CDF와 최종 q99는 기존 절차로 다시 적합한다. 공정 branch 원점수 및 보정 점수가 동일한지 검사한다. 임계값 변경으로 branch 경보 개수는 달라질 수 있음을 구분한다.
5. 테스트 전에 정상 FIT의 역할×phase 지원 수, pooled bank 보존, fallback 사용량, calibration finite/포화/q99 가능성을 검사한다. 지원 부족으로 phase bank가 사라져도 K/최소 지원/threshold를 낮추지 않는다. 설정·코드·입력 hash를 먼저 기록하고 정상 가능성 검사 후 테스트를 평가한다.
6. 실험 18과 전체 AUROC/AP·정상 오탐/이상 recall·GT 구간 탐지/지연을 비교한다. 관측/미관측별 외형 점수와 경보, pooled/phase bank 사용량, 추가/제거 경보를 함께 보고한다. 같은 전체 프레임과 고정 관측 mask로 비교하며 라벨은 평가에만 사용한다.
7. 단위 검사로 미관측의 phase bank 제외, pooled 단일 삽입, 최소 지원 fallback, 추론/보정의 동일 경로, 원본 phase 보존을 확인한다. 실제 데이터에서 pooled PCA·관측 mask·공정 점수 보존과 지표 재계산을 검증한다.

이 실험은 결측 관계가 있는 영상의 외형 조건화 가정을 검토한다. 올바른 객체를 복원하거나 semantic phase 정답을 얻는 방법이 아니며, pooled fallback은 공정 특이 정보를 잃을 수도 있다. R04는 반복 개발 장면이며 novelty·일반화 증거로 자동 해석하지 않는다. 결과 이후 추천순 3개를 새로 정하고 후속 전체 일정을 미리 확정하지 않는다.
