# 다음 실험 계획 — 18

상태: 실험 17 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

동일 객체 track에서 의미 점수를 시간적으로 집계하면 짧은 gate 변동과 관계 누락을 줄이면서 잘못된 anchor를 계속 차단할 수 있는가?

실험 17은 일부 정상 바이스 예시를 제외했지만, 정상 같은-track 연속 관측 1,868쌍에서 margin 부호 전환 190회를 기록했다. 새로 관계를 잃은 136개 구간 중 62개는 한 sample 길이였다. 테스트 체류 가용성은 41.83%→4.03%, Process AUROC는 0.6307→0.4638로 줄었다. 이것이 전부 flicker 때문이라고 단정하지 않고 gate 집계만 바꿔 검증한다.

## 고정할 변경

1. 실험 17의 CLIP text embedding·두 문장·margin>0 threshold·검출/crop/track 원본·R04 분할/seed를 유지한다. prompt·threshold·여러 window를 탐색하지 않는다.
2. role 1의 각 검출에 대해 **같은 track ID의 현재 및 직전 최대 2개 연속 sampled 관측 margin 중앙값**을 계산한다. window=3으로 고정하고 첫 1/2개 관측에서는 이용 가능한 관측만 사용한다. 같은 sample 안에서 미래 bbox를 섞지 않으며 영상마다 상태를 초기화한다.
3. 해당 track의 직전 관측 sample index와 현재 index 차이가 1보다 크면 history를 비운다. track ID가 바뀌면 새 history다. 누락 bbox 생성·gap 보간·미래 smoothing은 하지 않는다. 따라서 이미 관측한 후보의 판정만 안정화할 수 있다.
4. 집계 margin>0인 anchor 후보에 기존 area gate·track 우선 선택을 적용한다. 정상 관계 모델·PCA·전이·lognormal 체류를 동일 설정으로 재적합한다. appearance 후보는 유지하되 phase 변경으로 score가 달라질 수 있음을 명시한다.
5. 정상 FIT에서 원래/집계 margin 부호 전환, 후보·관계 관측, 한 sample 누락 구간, 여섯 정상 정성 사례, 군집·완결 체류 support를 확인한다. 지원 부족이면 실패로 기록하고 K나 최소 구간 수를 낮추지 않는다. 같은 track이 잘못된 바이스를 계속 추적하면 집계가 오히려 그 오류를 유지할 수 있다.
6. 지원이 충분하면 설정·코드·입력을 동결하고 동일 집계로 calibration/테스트를 변환한다. 정상 q99 가능성 검증 후 실험 17과 AUROC/AP·오탐/recall·구간 탐지/지연·branch 경보·관계/체류 가용성을 비교한다. 실험 16은 gate 없는 참고점으로 남긴다.
7. 시간 인과성, gap/track reset, 원본 배열 보존, 같은 sample 중복 track 처리, prefix 불변성을 검사한다. 개선이 있어도 semantic 역할 정확도·novelty·일반화를 자동으로 주장하지 않는다. 새 결과로 다음 추천순 3개를 갱신한다.

새 검출·언어 재생성·missing-phase 모델 교체·matched-support 대조는 이번 변경에 섞지 않는다. R04는 개발 장면이며 독립 최종 평가가 아니다.
