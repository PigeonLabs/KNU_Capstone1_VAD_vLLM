# 다음 실험 계획 — 38

상태: 실험37 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

외형 bank 선택 시 관계 phase를 한 번도 관측하지 못한 상태를 미확정으로 처리하면, 임의 초기 ID=0의 bank에 의존하는 경보를 피할 수 있는가? 첫 관측 이후의 hold 정책은 유지하여 오래된 누락을 처리하는 age 정책과 구분한다.

실험37 asinh hold의 첫 관측 전 test652 frames는 모두 정상이고 FP60이다. pool/age는 같은 구간에서 FP0이다. hold/age는 관측 중5,599 frames 및 짧은 누락1,725 frames에서 점수가 정확히 같지만, 첫 관측 전636 frames와 오래된 누락170 frames에서 다르다. 오래된 누락에서는 hold만TP16, age만TP12여서 hold 전체를 age로 교체하는 것과 초기 조건을 교정하는 것은 다르다. 정상 FIT/calibration에도492/80 frames의 미확정 구간이 있다. 이번 설계는 phase 관측의 정의를 교정하는 것이며 test에서 잘 나오는 시간 cutoff를 찾지 않는다.

## 변경 하나: 최초 관측 여부로 외형 요청 제한

- control은 실험37 asinh 모델의 hold/pool/age다. 변경군은 **현재까지 `relation_valid=True`를 한 번도 보지 못한 sample에는 외형 pooled bank(-1)를 요청**한다. 첫 유효 관계가 나타난 바로 그 sample부터 기존 경로별 정책을 그대로 적용한다.
- 이력은 영상별로 초기화하고 현재/과거 mask만 사용한다. 미래 첫 관측을 미리 찾거나, missing box/phase를 보간하거나, `phases` 배열을 -1로 덮어쓰지 않는다. 종료까지 관측이 없는 영상은 계속 pooled다.
- 공정 phase·전이·체류·same-pair gate는 그대로 유지한다. fit의 관측 기반 bank 표본/phase별 rank·PCA·관계 선택/descriptor/scaler/asinh 중심·모든 CDF 참조·support는 변경하지 않는다. full-normal dual-request CDF는 원래 경로 선택과 독립적으로 정의되므로 동일성을 검사한다. 정상 legacy appearance 참조가 요청 정책에 따라 달라지더라도 실제 사용 CDF와 구분해 기록한다.
- q99는 동일한 정상 calibration에서 각 요청 정책으로 다시 계산한다. 보정 후 경보 변화와 고정 control q99를 쓴 대조는 구분한다. threshold를 테스트에 맞춰 선택하지 않는다.
- pool/age는 이미 초기 pooled 동작을 하므로 변경군과 모든 출력의 정확한 동일성이 기대되는 음성 대조다. hold만 처음 관측 전 요청이 바뀌며, 첫 관측 후 원시 외형 점수·공정 점수는 같아야 한다. q99가 달라지면 이후 경보도 달라질 수 있어 점수와 경보 불변을 혼동하지 않는다.
- 새 phase 적합·검출·track·CLIP·VLM 호출 또는 llama.cpp 설정 변경은 없다. 새로운 age threshold나 시도별 추가 변형을 탐색하지 않는다.

## 정상 검증과 동결

1. 실험37의 정상 캐시·모델·보정 참조, 새 코드/계획을 먼저 동결하고 정상 단계에서 test/label 접근을 막는다.
2. 현재까지 관측 없음/첫 관측/재누락/영상 끝까지 누락/새 영상 reset, prefix·미래 불변을 검사한다. 초기 숫자 phase ID를 바꿔도 미확정 구간의 pooled 요청/외형 점수가 같아야 한다. 이는 전체 공정 grammar의 label permutation 불변 주장과 다르다.
3. 정상 FIT/PCA·공정·full-normal 요청 CDF 동일성과 control의 실험37 재현을 확인한다. 정상 calibration5영상 leave-one-video-out을 수행하고 held-out 영상은 보정에서 제외한다. 각 fold의 초기 구간 및 첫 관측 후 점수/경보, CDF/q99 변화, finite/bounded·support·q99<1을 기록한다.
4. 정상20/5영상의 첫 관측 시점·초기 sample/source-frame 수와 경로별 실제 bank 요청을 별도 집계한다. 기준은 `relation_valid`이며 track/역할 정확도를 새로 추정하지 않는다. 새 시각 판독이나 사례 재선정은 필요 없다.
5. 정상 결과·모델·q99를 동결한 뒤만 test를 진행한다. 기대한 pool/age 불변이나 hold의 첫 관측 후 점수 불변이 깨지면 원인을 먼저 검증하고 숨기지 않는다.

## 테스트와 판정

기존 R04 test19영상/8,154 frames에서 AUROC/AP, 정상 q99의 FPR/recall, 구간 gained/lost와 조건부 지연, Visual 대비 공정 기여를 보고한다. 초기 구간과 나머지 구간을 분리하고, 전체 경보 변화가 요청 변화인지 정상 q99 변화인지 확인한다. test에서는 관측 전652 frames가 모두 정상이라 초기 이상 탐지 능력을 입증하지 못하며, 이를 별도 한계로 명시한다.

이 변경은 임의 초기 상태의 사용을 막는 관측 기반 routing이다. 더 나은 일반화·novelty·역할 정확도나 FP60의 자동 제거를 미리 주장하지 않는다. 단일 R04 반복 개발 결과로 남기고, 완료 후 결과·의의·보완점·추천순3개를 보고/업로드한 다음 다음 변경 하나를 선택한다.
