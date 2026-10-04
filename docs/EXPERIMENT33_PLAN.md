# 다음 실험 계획 — 33

상태: 실험 32 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

체류 진입을 확인하지 못하고 관측이 끊기는 가장 직접적인 파이프라인 원인은 무엇인가? 정상 raw 후보 부재, semantic/면적 gate, 선택 track 교체를 구분할 수 있는가?

실험 32에서 FIT episode 257개 중 228개가 unknown-entry였다. 시작 원인은 재관측 156개, target 교체 45개, anchor 교체 24개, 영상 시작 3개다. 관측된 진입을 가진 29개 중 22개가 검열됐으며 관계 누락 11개, track 교체 9개, 영상 끝 2개였다. 현재 같은 쌍의 complete는 3/4개로 기존 support 10개에 미달한다. duration 분포를 복잡하게 만들기 전에 관측 손실의 원인을 확인해야 한다.

## 이번 구현과 고정 범위

- **정상 25영상의 관측 근거 추적 + 사전 표집한 정상 사례 검토**를 수행한다. 테스트 영상/특징/라벨/예측은 열지 않고 새 anomaly score·CDF/q99·duration 분포를 만들지 않는다.
- 실험 30 reset의 정상 캐시, 실험 18 고정 relation 모델·면적 gate·semantic 설정, 실험 17 text embedding, 실험 32 episode를 사용한다. VLM/검출기/CLIP 재호출이나 검출 threshold/semantic margin 변경은 없다.
- anchor/target 각각에 대해 sample별 raw 후보 수, 양의 면적 후보, semantic gate 통과 여부, 면적 gate 통과 여부, 두 조건의 교집합, 최종 선택 index/track을 저장한다. target에는 실제 코드에서 사용하지 않는 semantic gate를 새로 적용하지 않는다. anchor의 raw margin과 시간 median margin을 구분한다.
- 순차 filter와 두 조건의 교집합을 명시해 이중 집계하지 않는다. 역할별로 후보 자체 부재, 유효 면적 부재, semantic 배제, 면적 배제, 교집합 부재, 후보 존재를 구분한다. 어느 역할의 실패로 relation이 invalid가 되었는지 두 역할의 조합도 기록한다. 합계와 구현의 실제 선택/valid가 일치해야 한다.
- 현재 prior-track 선호·confidence 선택을 재현하고 track 교체가 이전 ID 후보의 소실인지, 해당 후보의 filter 탈락인지, 새 후보 선택인지 추적한다. ID 교체를 실제 객체 교체 또는 tracking 오류로 단정하지 않는다.
- 실험 32의 검열 경계·unknown-entry 재관측/교체 시작을 원인 trace에 연결한다. 누락 구간의 길이와 직전·재관측 선택 쌍 ID도 기록하되 동일 ID 재등장을 이용해 과거 누락을 채우거나 진입을 만들어 내지 않는다.

## 정상 시각 검토

검토 전에 metadata만으로 다음 네 그룹에서 **각 최대 6개** 사례를 동결한다: (1) 관측된 진입 후 relation_missing으로 검열된 경계, (2) 재관측으로 시작한 unknown-entry, (3) track 교체로 시작한 unknown-entry, (4) 같은 쌍으로 완결된 control. 각 그룹은 영상별 round-robin, sequence/source frame 순으로 결정한다. 중복 경계가 있다면 같은 사례를 재사용하되 그룹 소속을 모두 명시한다. 검토한 뒤 유리한 사례만 다시 고르지 않는다.

선택 경계의 이전/현재/다음 sampled frame을 정상 원본에서 확인한다. 다음 frame은 사후 진단 자료이며 추론 입력이 아니다. bbox/role/track/margin을 함께 보되 다음을 구분해 메모한다: 화면 내 객체/가림 여부의 관찰, 선택 bbox가 가리키는 것으로 보이는 대상, 크기/역할 혼동 가능성, 판단 불가. 검출 recall·역할 정확도·의미 GT로 정량화하지 않는다. 원본 이미지/접촉 시트는 로컬에만 두고 공개 보고서는 집계와 사례 metadata/메모만 포함한다.

## 검증·보고

1. 코드·설정·계획·정상 feature/relation 모델·기존 정상 점수 hash를 동결한다. 후보 0개·면적 0/초과·semantic 임계값 경계·교집합 부재·선택 유지/교체·누락을 검사한다.
2. 정상 25영상 모든 sample에서 원래 valid/선택/phase/descriptor/margin을 재현한다. 각 역할의 단계별 후보 수와 최종 실패 조합이 relation mask를 정확히 설명해야 한다. source frame/episode 연결과 그룹 표집 hash를 확인한다.
3. FIT/calibration을 나눠 역할·filter 원인·관측/중단/재관측/track 교체 분포를 보고한다. 원인별 누락 구간과 검열/unknown-entry의 연결을 제시하되 raw 후보 부재를 곧바로 false negative라고 부르지 않는다.
4. 시각 검토는 목적 표집의 보조 근거로 보고하고 불확실한 경우 null/판단 불가를 유지한다. 모델 개선 방향은 전체 정상 분포와 검토 근거를 함께 보고 선택한다.
5. 모델/점수/정상 입력 불변 및 test 접근 없음, 결과·의의·보완점·추천순 3개를 검증하고 GitHub에 업로드한다. 실제 detector/역할/추적/체류 변경은 이 결과를 본 뒤 하나를 선택한다.

## 한계

후보의 부재 원인은 카메라 가림·실제 부재·검출 실패를 포함할 수 있어 GT 없이 분리되지 않는다. semantic margin과 track ID는 정답이 아니며 목적 표집으로 정확도를 추정할 수 없다. R04 정상 25영상, 단일 split/seed, 독립 녹화 그룹/FPS/action GT 부재를 유지한다. 진단 기능 구현이나 원인 비율만으로 신규성·검출 개선을 주장하지 않는다.
