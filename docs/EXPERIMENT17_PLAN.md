# 다음 실험 계획 — 17

상태: 실험 16 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

검출기가 부여한 역할을 그대로 믿는 대신, 의도한 객체와 정상 영상에서 확인한 혼동 객체의 언어 표현을 대조하면 잘못된 관계 anchor의 영향을 줄일 수 있는가?

실험 16은 경보 불능을 해소했지만 체류 없는 기준선과 경보가 동일하고 새 이상 구간을 찾지 못했다. 실험 15에서 정상 예시 6개 중 5개의 `lid` bbox가 바이스를 가리킨 문제도 그대로다. 다음은 길이 점수의 추가 조정 대신 **관계 모델에 들어가는 객체 역할의 의미 확인 단계**를 도입한다.

## 고정할 변경

1. R04 정상 분할·seed·원본 검출/crop 특징/track은 실험 16과 동일하게 재사용한다. 기존 frozen CLIP ViT-B/32의 crop embedding과 같은 체크포인트의 text encoder를 사용한다. 새로운 anomaly 라벨이나 추가 학습은 사용하지 않는다.
2. 정상 정성 검토의 의도한 금속판과 혼동 바이스를 근거로 다음 두 문장을 고정한다. positive: `A photo of the hinged metal cutting blade of a manual shear.` negative: `A photo of a bench vise fixed on a workbench.` 정상 crop의 margin을 보기 전에 문장을 고정하며 test 결과로 문장을 반복 수정하지 않는다.
3. L2 정규화한 text/crop embedding의 cosine으로 `margin = similarity(positive) - similarity(negative)`를 계산한다. **margin > 0인 role 1 bbox만 관계 anchor 후보로 허용**한다. 0/음수는 후보에서 제외한다. confidence threshold·margin threshold 탐색은 하지 않는다.
4. 이 gate는 관계 상태용 anchor에만 적용한다. 원래 bbox/crop/track·appearance 후보·판재 target 후보는 보존한다. 이후 기존 normal area gate·track 우선 선택·관계 descriptor를 적용하고, FIT 관계 KMeans K=4·PCA·전이·lognormal 체류는 바뀐 정상 phase에 맞춰 재적합한다. 따라서 최종 Visual 점수는 입력 crop이 같아도 phase 변경으로 달라질 수 있다.
5. 먼저 정상 FIT에서 anchor 후보 전후 수, 유효 관계, 기존 6개 정상 확인 사례의 gate 결과, 군집/완결 체류 support를 기록한다. gate가 바이스를 줄이는 동시에 실제 금속판을 누락할 수 있고, 누락된 진짜 bbox를 새로 복원하는 모듈은 아니다. margin의 부호를 검출 정확도로 간주하지 않는다.
6. 수치상 정상 관계/체류 support가 기존 최소 조건을 충족하지 않으면 적용 실패로 보고한다. K·최소 support·문장을 결과에 맞춰 바꾸거나 미지원 모델을 성공한 실험으로 대체하지 않는다. 정상 사례의 의미 확인은 보조 정성 검토이며 전체 bbox GT가 아니다.
7. 지원이 충분하면 설정·문장·text embedding·코드·입력 출처를 동결한 뒤 동일 gate로 calibration/테스트 관계를 변환하고 실험 16과 비교한다. 정상 q99 경보 가능성, AUROC/AP·오탐/recall·구간 탐지/지연, 새 누락·관계/체류 가용성을 함께 평가한다. 원본 입력 보존과 시간 인과성도 검증한다.

문장의 객체 해석은 정상 영상 확인에서 나온 수작업 의미 지식임을 공개한다. 이를 완전 자동 vocabulary 발견이나 학습 없는 새로운 기본 모델로 주장하지 않는다. R04는 개발 장면이며 표준 CLIP 유사도 자체의 novelty도 주장하지 않는다. 다음 추천순 3개는 실험 17의 실제 결과로 갱신한다.
