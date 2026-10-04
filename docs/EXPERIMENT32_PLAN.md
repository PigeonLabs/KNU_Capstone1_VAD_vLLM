# 다음 실험 계획 — 32

상태: 실험 31 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

객체 교체·누락·영상 끝을 실제 phase 종료와 구분하면, 정상 체류 자료는 어느 정도 남으며 어떤 맥락의 지속 시간을 학습할 근거가 있는가?

실험 31에서 전이의 독자 경보는 0/0이 되었고 공정 추가 구간 1개는 체류에서만 남았다. 정상 감사에서 현재 FIT 1→3 14 runs/10영상, 3→1 10 runs/6영상 중 같은 선택 쌍으로 진입·유지·종료한 것은 3 runs/2영상, 4 runs/3영상뿐이었다. 기존 최소 10 complete runs 규칙을 그대로 적용하면 두 맥락 모두 지원할 수 없다. 단순히 threshold를 낮추거나 관측 중단을 종료로 바꾸기 전에 학습 자료의 관측 근거부터 바로잡는다.

## 이번 구현 범위

**Normal-only episode 생성 및 지원성 검사**를 수행한다. 새 duration 분포·anomaly score·q99는 적합하지 않는다. 실험 31의 기존 모델·점수는 유지한다. 이번 결과에서 새 AUROC/AP·탐지 성능을 주장하지 않고, 학습 가능한 사건/검열 자료의 수·영상 분포·관측 길이와 실패 이유를 보고한다.

- 실험 30 reset의 R04 정상 FIT 20/calibration 5개 feature/선택/phase를 입력으로 고정한다. 검출·track·VLM·CLIP·phase 중심은 변경하지 않는다. 테스트 영상·feature·라벨은 열지 않는다.
- 같은 쌍의 연속 관측에서 phase가 바뀌면 새 phase의 **관측된 진입**으로 기록한다. 이전 phase를 entry context로 저장한다. 실제 action 진입 시점 GT를 뜻하지 않는다.
- 이후 같은 쌍이 유지되고 다음 phase로 바뀌는 outgoing edge도 같은 쌍이면 **관측된 완결**로 기록한다. 표본 간 간격의 한계가 있는 sampled-phase duration이며 실제 연속 시간의 정확한 종료 시각으로 주장하지 않는다.
- 진입은 관측했으나 객체 쌍 교체·missing·영상 끝 때문에 같은 episode를 더 추적할 수 없으면 **우측 검열**로 기록한다. 원인은 anchor/target/both 교체, 관계 누락, 영상 종료로 구분한다. 실제 phase가 끝났다고 간주하지 않는다.
- 검열된 길이의 보수적 하한은 `마지막으로 같은 쌍·phase가 관측된 source index − 관측 진입 index`로 기록한다. 다음 missing/교체 frame까지 지속했다고 채워 넣지 않는다. 관측 중단 frame은 별도 metadata이며 실제 종료 시각 또는 종료 시각의 상한으로 사용하지 않는다. 0 길이 하한은 허용·별도 집계하고 로그 분포에 임의 epsilon으로 넣지 않는다.
- 영상 시작, 재관측, 객체 교체로 시작한 구간은 진입이 확인되지 않은 **unknown-entry**로 구분한다. 같은 쌍의 phase 변경이 나타나기 전까지 새 진입 시각/맥락을 만들어 붙이지 않는다. 그 관측 구간 길이를 완전한 duration 정답으로 사용하지 않는다.
- 관측 sample은 episode 출처를 추적할 수 있게 배정하되 outgoing phase-change sample은 새 phase episode에 귀속한다. missing sample은 별도 미관측 항목이다. episode 종료 원인·경계 source index·track 쌍·맥락·마지막 관측·완결/검열 여부를 기계 판독 형태로 보존한다.

## 검증과 보고

1. 구현/계획/정상 입력/기존 정상 모델·점수의 hash를 감사 전에 동결한다. complete, track 교체, missing/reacquisition, 영상 시작/끝, unknown-entry, 0 하한, 양 역할 교체와 경계 중복/누락을 테스트한다. 표본의 귀속 합계와 valid/missing 합계가 원자료와 일치해야 한다.
2. 같은 쌍의 완결 run은 실험 31 provenance 감사의 1→3 3개/2영상, 3→1 4개/3영상 및 개별 start/end/length와 일치하는지 독립 재구성한다. 불일치는 규칙 차이나 구현 오류로 설명하고 숨기지 않는다.
3. normal FIT와 calibration을 분리해 맥락별 complete/right-censored/unknown-entry 수, 고유 영상 수, 양의 관측 하한과 0 하한, 종료 원인, 길이 분포·영상 집중도를 보고한다. calibration을 FIT support에 합치지 않는다.
4. 인과적 prefix 검증에서는 영상이 임시로 끝나며 생긴 검열 기록과 아직 진행 중인 episode가 달라질 수 있음을 명시한다. 과거에 관측된 진입·track 쌍·경계 이전 상태가 미래 입력에 의해 바뀌지 않는지 검사한다. 완료 여부를 미래 정보로 과거 inference score에 채워 넣지 않는다.
5. 검열이 역할 오류/가림/실제 공정 상태와 연관될 수 있어 독립적인 관측 중단이라고 가정할 수 없음을 보고한다. duration 생존 모형이나 다른 분포가 적합하다는 결론을 개수만으로 내리지 않는다.
6. 기존 정상 모델/점수 불변, test 접근 없음, 결과 표/필요한 집계 그래프, 의의·보완점·추천순 3개를 검증하고 GitHub에 업로드한다. 다음 모델 변경은 이번 episode 분포와 지원성을 본 뒤 결정한다.

## 한계

track ID는 실제 identity 정답이 아니며 동일 ID 역할 오류는 남는다. 관측된 phase 진입/종료도 latent cluster 경계다. stride 4와 FPS/timestamp 부재로 실제 초 단위 지속 시간은 알 수 없다. 정상 25영상은 작고 녹화 그룹 독립성이 확인되지 않았다. 검열을 따로 기록하는 구현만으로 신규성·검출 성능 향상을 주장하지 않는다. 논문은 넓은 파이프라인 구현 범위를 유지한다.
