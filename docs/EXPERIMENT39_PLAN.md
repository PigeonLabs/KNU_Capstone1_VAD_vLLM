# 다음 실험 계획 — 39

상태: 실험38 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

체류 점수를 공정에 결합할 때, 실제 phase 진입 이후 같은 객체 쌍이 유지됐다는 근거를 요구하면 무엇이 달라지는가? 분포 재학습과 분리해 관측 근거의 정합성과 경보/순위 변화를 검증한다.

실험38은 외형의 초기 미확정 상태를 교정했지만, 체류 가용1,889 test frames 중144는 진입 이후 동일 쌍 근거가 없다. 기존 정상0→1/1→2 학습 지원13/16과 strict complete9/4의 차이도 남았다. 다만 현재 체류 독자 FP13/TP19는 모두 동일 쌍 진입 근거를 가진 영상08의1→2다. 이번 gate가 이를 없애거나 새 구간을 탐지할 것이라고 기대 효과를 과장하지 않는다.

## 한 가지 변경: 체류 점수 결합 gate

- control은 실험38 guarded hold/pool/age다. 변경군은 기존 `dwell_valid`에 추가로 **phase 진입 이후 현재까지 동일한 선택 anchor/target track 쌍이 연속 관측됨**을 요구해 체류를 process max 결합에 사용한다.
- 진입은 바로 앞/현재 sample 모두 관계가 관측되고, phase가 바뀌며, 두 sample의 선택 쌍이 같은 경우에만 확정한다. 최초 sample·재관측 시 자동 진입을 가정하지 않는다. 누락 또는 어느 객체 track이든 바뀌면 근거를 해제한다. 같은 쌍을 다시 찾더라도 새로운 관측 phase 진입 전에는 회복하지 않는다. 같은 쌍의 다음 phase 전환은 그 새 진입으로 갱신한다.
- 현재/과거 관측만 사용하고 미래 이탈 여부를 보지 않는다. 영상마다 상태를 초기화한다. 이를 실제 물리적 identity/동작 GT로 간주하지 않는다.
- **기존 체류 분포·원시 점수·`dwell_valid`는 보존**하고, 결합용 근거 mask/가려진 체류 점수를 별도로 기록한다. 최소 지원10, lognormal 파라미터, 정상 CDF, 전이 gate/점수, 외형 점수·초기 observation gate는 그대로다. 변경된 process/combined로 정상 q99만 다시 계산한다.
- 객체 선택·feature cache·relation descriptor·asinh phase 모델·phase별 PCA rank·bank·normal FIT/보정 분할은 고정한다. 새 VLM/검출/encoder 호출이나 llama.cpp 설정 변경은 없다. 학습 표본을 strict complete로 교체하거나 지원 기준을 낮추는 변경을 동시에 하지 않는다.

## 정상 검증과 동결

1. 기존 정상 모델/점수/특징과 새 코드·계획을 동결하고 normal 단계에서 test/label을 차단한다.
2. 관측된 같은 쌍 phase 진입, 진입 시 교체, phase가 같은 채 교체, 누락/재관측, 새 영상, prefix·미래 비참조를 검사한다. 기존 전이 same-pair mask의 한 step 검사와 누적 체류 이력을 혼동하지 않는다.
3. 정상 FIT에서 control의 PCA·전이·체류·실제 CDF를 정확히 재현한다. gate를 켜도 이 파라미터와 원시 외형/전이/체류가 같아야 한다. process/combined는 원래 값보다 커질 수 없고, 근거가 있는 sample에서는 원래 점수와 같아야 한다. q99 변화로 최종 경보가 늘 가능성은 별도로 검사한다.
4. 정상 calibration5영상 leave-one-video-out으로 held-out을 보정에서 제외하고 finite/bounded·기존 support·q99<1을 검증한다. gate 전후 사용 가능한 체류 samples/frames, 차단 원인, 영상별 정상 경보와 q99를 기록한다. 지원 부족을 숨기지 않는다.
5. 정상 모델·보정·근거 audit를 동결한 뒤 test를 연다. 새 사례 이미지 판독이나 phase 재학습은 필요 없다.

## 테스트와 판정

기존 R04 test19영상/8,154 frames에서 AUROC/AP, 자신의 정상 q99의 FPR/recall, 구간 gained/lost·조건부 지연, Visual 대비 공정 독자 기여를 보고한다. 변경군 점수에 고정 control q99를 적용한 대조로 gate 효과와 재보정 효과를 분리한다. 근거 없는 체류의 제거와 기존 FP13/TP19 유지 여부를 검증하되, test 정보를 gate cutoff나 파라미터 선택에 쓰지 않는다.

gate는 학습 분포의 관측 정의를 교정하지 않으며, 같은 쌍에서도 틀린 역할과 비정상적인 정상 CDF가 남을 수 있다. 정상 support13/16과 strict complete9/4의 불일치도 계속 보고한다. 경보가 그대로여도 근거/순위 변화와 효용의 한계를 결과로 남긴다. 단일 R04 반복 개발로 일반화·통계적 유의성·novelty를 주장하지 않는다. 완료 후 결과·의의·보완점·추천순3개를 업로드하고 다음 변경 하나를 선택한다.
