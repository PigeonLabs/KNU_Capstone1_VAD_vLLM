# 실험41 정상 적합 중단과 명시적 dwell unavailable 처리

테스트 특징·라벨에 접근하기 전 정상 모델 적합이 R04에서 중단됐다. R01~R03의24개 full 모델 계산 뒤, R04의 `NormalDwell.fit`이 `No supported normal dwell state`를 발생시켰다. 실패 로그·부분 모델은 local `artifacts/experiment41/normal_fit_attempt1/`, 당시 protocol·driver·R04 config는 [실패 시점 기록](../results/experiment41/normal_fit_attempt1/)에 보존한다. [복구 기록](../results/experiment41/normal_fit_recovery.json).

R04 FIT20영상에서 observed-entry/exit complete run은 상태0/1/3별5/5/7개이며 상태2는0개다. Entry 문맥은1→3:5,3→1:4,3→0:4,0→3:2,1→0:1,0→1:1개다. 모두 사전 최소 지원10 미만이다. 이 값은 legacy complete-run 정의이고 동일 객체 쌍이 검증된 identity GT의 개수가 아니다.

**원래 strict 파이프라인의 R04 적합 실패 자체가 실험 결과다.** 이후 정량 평가는 정상 자료만 보고 다음의 명시적 availability 정책을 추가한 경로다. Detector가 유일한 학습 변경이라는 범위는 유지하지만, runtime 처리까지 완전히 같은 파이프라인이라고 표현하지 않는다.

- 최소 지원10과 기존 완결·관측 정의를 유지한다. 미지원 duration을 만들어 채우거나 검열 하한을 완결로 바꾸지 않는다.
- R04 config에 `dwell_allow_unavailable: true`를 추가한다. 새로운 optional subclass만 사용하며 기존 strict classes와 실험40 코드는 수정하지 않는다.
- 충분한 상태·문맥이 있으면 기존 lognormal fit/score를 그대로 호출한다. 기존 경로와 배열 단위 일치를 테스트한다.
- 어떤 문맥도 충분하지 않으면 `available=False`, 이유 `no_supported_normal_entry_context`, 빈 context distribution/log parameters/reference를 기록한다. `dwell_valid=False`이므로 체류 증거를 결합하지 않는다. 0으로 저장되는 dwell score는 정상 판정값이 아니라 masking용 값이다.
- 외형·전이 계산과 그 유효성 gate는 유지하고, 정상 CDF/q99와 holdout 전체를 다시 계산한다. Detector/visual 가중치와 이미 추출한 정상 features는 변경하지 않는다.
- 모델 동결과 테스트 점수 동결을 다시 수행한 후에만 라벨을 평가한다. 중단 전 테스트 결과로 복구 정책을 선택하지 않았다.

새 테스트는 빈 완결 자료, pooled state만 충분하고 entry context는 부족한 경우, 지원되는 경로의 strict 구현과 정확한 일치 및 재적합 시 상태 초기화를 검증한다. 이 처리가 부족한 공정 관측이나 phase 편중을 해결하지는 않으며, R04 체류 이상은 이번 learned 경로에서 탐지 근거로 사용할 수 없다.
