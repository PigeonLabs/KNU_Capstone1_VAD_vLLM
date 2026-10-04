# 다음 실험 계획 — 31

상태: 실험 30 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

관계가 연속 관측되더라도 선택 객체 쌍이 다르면 이전/현재 phase를 동일 대상의 전이로 비교해도 되는가? 실험 30에서 평균 이력 혼합을 제거했지만 calibration 02의 frame 76/252에 새로운 정상 2→1 전이 경보가 생겼다. 두 곳 모두 anchor track 교체 경계이며 나머지 calibration에는 해당 edge가 없었다. 이전 상태 reference 31개가 있어도 점수는 1이다. 테스트에서 전이의 독자 경보는 여전히 FP 16/TP 4이고 독자 추가 구간은 없었다.

## 이번 변경과 고정 조건

1. 실험 30 reset의 세 외형 경로를 기준으로 **전이 점수 결합 gate만** 바꾼다. `valid[t-1] & valid[t]`에 선택된 anchor 및 target의 track ID가 각각 이전 sample과 같다는 조건을 추가한다. 영상 첫 sample, 누락 또는 쌍 교체 경계에서는 전이 결합 값을 0으로 한다. 재관측 직후는 기존 gate도 닫힌다. 객체 선택·track을 수정하거나 미래 정보를 보지 않는다.
2. 실험 30의 reset relation 특징/phase·중심·normal FIT 표본/PCA/common rank·전이 확률/allowed grammar·full-transition CDF 모집단과 reference·체류 분포 및 가용성을 고정한다. 관측 전이 CDF를 재도입하지 않는다. 체류는 이번에 바꾸지 않으며, 체류 entry가 identity 경계를 가로지를 수 있다는 남은 문제를 명시한다.
3. 외형 hold/pool/age(τ=56)를 모두 유지한다. 기준 `30_reset_hold/pool/age`와 새 `31_hold/pool/age`를 비교한다. 세 외형 경로 중 하나를 test 성능으로 선택하지 않는다.
4. Combined 정상 q99는 새 gate를 반영해 calibration에서 다시 계산한다. 점수/CDF 자체가 같아도 q99는 달라질 수 있으므로 own q99와 paired baseline q99에서의 사후 진단을 둘 다 보고한다. 임계값 조정으로 정상 feasibility 실패를 숨기지 않는다.

## 검증 순서와 판정

- 코드/설정/계획·기준 캐시/모델 hash를 정상 적합 전에 동결한다. 두 역할 중 하나의 교체, 두 역할 모두 유지, 누락·재관측, 첫 sample, prefix를 테스트한다. 선택 detection index의 유효성과 배열 정렬을 검증하고 invalid=-1이 마지막 detection으로 잘못 인덱싱되지 않게 한다.
- 정상 FIT/calibration 25개 영상에서 새 gate가 기존 gate의 부분집합임을 확인한다. 열린/닫힌 전이를 영상·latent edge·anchor/target 교체로 나눠 보고한다. 정상만으로 6구성×5 holdout과 full calibration feasibility를 검사한다. 외형/원래 transition percentile/체류 점수·가용성/정상 reference가 기존과 같은지 배열 단위로 재현한다.
- 정상 조건을 통과한 새 구성만 R04 test 19개에 한 번 적용한다. 파이프라인 입력·점수 생성 뒤 라벨을 읽는다. detector/VLM/CLIP 재호출은 필요 없다.
- AUROC/AP, own q99 FP/TP·26구간 탐지·조건부 지연, 전이/체류 독자 기여, 제거된 신호의 정상/이상 프레임과 구간을 보고한다. 단지 두 정상 반례가 사라진 것을 전체 개선으로 부르지 않는다.
- q99가 바뀌면 동일 새 점수에 기준 q99를 적용하여 gate 효과와 threshold 효과를 분리한다. q99 불변이면 닫힌 전이 밖 점수 불변을 확인한다. 기존에 높았던 전이가 사라져도 max의 다른 branch가 경보를 유지할 수 있으므로 transition 제거와 최종 경보 제거를 구분한다.
- 결과·의의·보완점·추천순 3개를 보고하고 GitHub에 업로드한다. 이후 변경은 이 결과에서 다시 결정한다.

## 한계

track ID는 검증된 identity가 아니다. 동일 객체의 track 단절, 실제 제품 교체 등에서도 gate가 닫혀 유효한 공정 변화/이상을 놓칠 수 있다. 같은 ID를 유지하는 역할 오류는 그대로 남는다. 새 3→1 체류 지원은 10 runs/6개 영상뿐이고 체류와 전이의 identity 정책은 이번 대조에서 달라질 수 있다. 이를 이번에 같이 바꾸면 원인 분리가 어려워 별도 후보로 남긴다. 반복 R04 개발·단일 seed·의미 GT/FPS/독립 녹화 그룹 부재를 유지하며 novelty나 일반화를 점수만으로 주장하지 않는다.
