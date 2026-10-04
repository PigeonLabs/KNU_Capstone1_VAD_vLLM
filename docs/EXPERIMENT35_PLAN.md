# 다음 실험 계획 — 35

상태: 실험 34 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

eligible anchor를 semantic margin만으로 즉시 교체하는 대신, 대안의 연속적인 우위를 확인하면 역할 근거와 시간적 연속성의 상충을 줄일 수 있는가?

실험 34에서 연속 valid 쌍 교체가 정상 FIT 69→135, calibration 16→37회로 늘고 strict complete는 7→6, 5→2개로 줄었다. semantic의 anchor 교체 중 기존 후보도 eligible인 자발적 교체는 36/11회, 한-sample 뒤 이전 ID로 돌아온 경계는 38/9회다. 최종 AUROC/AP·구간 탐지는 모두 낮아졌다. 이에 **자발적 교체에 2회 연속 우위 확인을 추가하는 한 변경**을 대조한다.

## 정확한 변경 규칙

대조군은 실험 34의 즉시 semantic 우선 선택이다. 변경군은 anchor 선택에 다음 상태를 추가한다. target 선택·후보 집합·면적/semantic gate·margin causal median은 고정한다.

1. eligible anchor가 없으면 미선택하고 대안 확인 상태를 비운다. 기존 prior ID는 기존 구현처럼 유지한다.
2. 이전 선택 ID가 현재 eligible에 없으면 현재 semantic 최대를 즉시 선택하고 대안 확인 상태를 비운다. 존재하지 않거나 탈락한 bbox를 만들어 유지하지 않는다.
3. 이전 선택 ID가 eligible이면 현재 semantic 최대 후보를 계산한다. 동점은 기존 track→confidence→작은 cache index로 정하며, 이전 track이 최대이면 그대로 유지하고 대안 확인 상태를 비운다.
4. 다른 ID가 최대이면 그 ID가 **직전 sampled step에서도 같은 대안으로 최대였는지** 확인한다. 같으면 확인 횟수를 늘리고, 다르거나 sample이 끊기면 1부터 시작한다. 2회가 되면 대안으로 바꾸고 확인 상태를 비운다. 1회만 확인됐으면 현재 eligible 이전 anchor를 유지한다.
5. 확인 상태는 영상/transform 호출마다 새로 시작한다. 미래 sample은 사용하지 않는다. target 미관측에서도 anchor가 관측됐다면 역할별 선택·확인 상태는 갱신한다. 2라는 값은 이번에 사전 고정하며 다른 값 탐색이나 test 기반 선택을 하지 않는다.

신규 hysteresis의 대상은 기존 후보가 유효한 자발적 교체뿐이다. 이전 후보 부재 57/16회와 filter 탈락에 따른 강제 교체는 그대로 남는다. 2회 확인은 원래 stride 4의 연속 관측 확인이며 고정된 실제 시간 길이로 해석하지 않는다.

## 정상 재현·적합·평가

- 정상 FIT 20/calibration 5영상, seed 42, R04 기존 데이터·특징을 유지한다. 실험 34 semantic 캐시를 대조군으로 정확히 재현한다. 원본 detector/CLIP/VLM 재호출이나 서버 설정 변경은 없다.
- 후보 eligibility, margin, target 선택, valid mask는 두 구성에서 동일해야 한다. pending challenger 교체/소실·동점·누락·기존 ID 미가용·target 누락·새 영상·prefix/미래 불변성을 검증한다.
- 선택된 쌍 변경 시 평균 reset, 실험 18 고정 관계 중심/scale, 실험 31 같은 쌍 전이 결합, 기존 체류 학습/점수 정책, hold/pool/age fallback 및 τ=56은 유지한다. descriptor/phase는 각 선택으로 재계산한다.
- 정상 FIT로 외형 PCA/전이/체류를 다시 적합하고 공유 phase bank rank는 양 구성 자연 rank와 실험 34 상한의 최소로 맞춘다. 표본/지원·새 bank/소실은 별도 보고한다. 공통 rank가 달라지면 단순히 실험 34 점수를 재사용하지 않는다.
- 정상 calibration CDF/q99 및 leave-one-video-out 5개 fold를 다시 적합한다. finite/bounded·support·q99<1을 검증하고 실패한 구성은 실패로 남긴다. 정상 FIT·calibration을 합치거나 test 라벨로 고치지 않는다.
- 동일한 24개 정상 경계의 전후 선택을 검토한다. 자발적/강제 교체, 한-sample 왕복, 확인 지연, 선택 margin, strict complete/censored/unknown-entry를 기록한다. role/action GT 없이 정확도로 계산하지 않는다.
- 정상 단계·설정·코드·시각 검토를 동결한 뒤 기존 R04 test 19영상/8,154 frames를 평가한다. 세 fallback 모두 AUROC/AP, 자신의 정상 q99에서 FPR/recall·구간 탐지/조건부 지연, 외형 대비 공정 독자 기여를 보고한다. control q99를 같은 변경군 점수에 적용한 대조는 사후 진단으로만 사용한다.

## 판정과 한계

교체 수 감소만으로 성공으로 판단하지 않는다. 올바른 역할로 복구가 늦어지거나 잘못된 바이스를 더 오래 유지할 수 있다. pending 규칙이 실제 잡음보다 필수적인 객체 변화를 억제하는지도 확인한다. missing이나 후보 탈락은 해결되지 않으며 고정 관계 중심·기존 체류의 ID 미검증 한계는 남는다.

R04는 반복 관찰한 개발 장면이다. 독립 녹화 그룹·bbox/action GT·FPS 부재, 단일 split/seed, 원본 라벨 정렬 한계를 유지한다. 이 규칙을 기존 연구 대비 novelty나 논문 주제로 미리 확정하지 않는다. 완료 후 결과·의의·보완점·추천순 3개를 갱신해 업로드하고 다음 변경 하나를 정한다.
