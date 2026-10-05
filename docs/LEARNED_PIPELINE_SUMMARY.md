# 모듈별 학습 단계 종합 — 실험40~43

요청한 공유 시각 표현과 공정별 모델 분리를 구현하고, 각 단계의 학습·평가·결과 보고를 완료했다. 결과에 따라 다음 한 요소만 선택했다. **모든 요소를 학습하면 성능이 순차적으로 상승한다는 결과는 얻지 못했다.** 정상 학습 loss, 약한 표적 일치, AUROC, 경보 recall은 별도 기준이었다.

| 단계 | 실제 학습한 요소 | 공유/분리 방식 | 확인한 결과 |
|---|---|---|---|
| [40](EXPERIMENT40.md) | Frozen CLIP / frozen MobileCLIP2-S2 / MobileCLIP2 LoRA / visual tower full FT | R01~R04 공유 encoder; 공정별 통계 정상 모델 | Visual C(LoRA) 평균 Combined AUC0.6702, full FT0.6354. 3seed 관찰에서 LoRA가 높았지만 모든 학습 설정의 일반적 우열로 주장하지 않음 |
| [41](EXPERIMENT41.md) | GroundingDINO decoder·bbox head partial FT,11,187,460변수 | 공통 detector,40의8개 visual arm 동결 | 정상 val loss는3.7955→0.4871. C AUC0.6702→0.6421, recall17.01→6.93%로 낮아져 box 교정이 전체 탐지를 보장하지 않음을 확인 |
| [42](EXPERIMENT42.md) | CLIP 위 rank8 association metric adapter,8,192변수 | 공통 head; 지원되는 R04 material만 appearance 연결 사용 | Val objective0.1848→0.0365. 정상1/test2 연결 추가, C AUC0.6418. FPR/recall/event 변화 없음. 다른 역할은 IoU fallback |
| [43](EXPERIMENT43.md) | Process별 linear 및 rank8 residual adapter phase head,총8개 | Shared frozen 시각 특징 + R01/R02/R03/R04 별도 head | C AUC: teacher0.6744 / linear0.6602 / adapter0.6672. Adapter의 weak agreement가 높아도 teacher 대조를 일관되게 넘지 못함 |

Transition/dwell/appearance subspaces/CDF/q99는 공정별 정상 자료로 적합했다. 이 통계 모델에 LoRA를 적용하지 않았다. 새로운 phase 분할에서 R04의 일부 dwell context 지원이 회복됐지만 물리적 동작 정확도나 모든 상태의 지원 회복으로 해석하지 않았다. Foundation VLM을 추가 fine-tuning하지 않았으며40~43의 해당 단계에는 원격 유료 API가 필요하지 않았다.

## 비교 범위

정상111영상은 representation train70/val19/calibration22로 분리했다. Downstream 정상 분포 FIT에는 앞의70+19를 사용한다. 테스트66영상의 strict valid31,550프레임·66events를 동일하게 사용했다. R02 길이 불일치1,912프레임은 전체 unknown으로 제외하고 원본 라벨 위치·FPS를 추정해 메우지 않았다. 독립 녹화 그룹 검증이 없고 반복 관찰한 개발 실험이다.

각 단계의 정상 약지도와 모듈 변경이 다르므로 서로 다른 단계의 수치만으로 알고리즘 기여를 단정할 수 없다. 41에는 검수한 box 약지도,42에는 Codex가 검수한 일부 negative pair,43에는 train에서 적합한 weak geometric target이 추가됐다. Anomaly label은 학습이나 checkpoint 선택에 사용하지 않았지만, 전체를 무주석 학습이라고 표현하지 않는다.

각 단계는 train/normal/test-score checkpoint와 source hash를 보존하고 test score를 고정한 뒤 지표를 계산했다. 43에서는190개 테스트,96 full+528 holdout,1,584 test 예측과576개 AUROC/AP 독립 계산을 검증했다. 학습 weight·feature cache·원본 영상은 로컬 ignored artifacts에 있고 GitHub에는 코드·설정·결과·보고서·그래프만 업로드했다.

## 논문에 사용할 수 있는 결론

현재 기여는 **모델 특성에 맞는 적응 강도를 선택하는 모듈별 파이프라인 구현과 대조 실험**이다. 공유 encoder와 detector를 개별 적응시키고, 소규모 association adapter 및 공정별 phase head를 학습하며, 정상 공정 통계의 지원 부족을 명시적으로 다루는 과정을 재현 가능하게 기록했다. Teacher 변화와 head 학습을 구분한 점, 실패와 경보 trade-off를 함께 확인한 점을 구현·실험 기여로 설명할 수 있다.

이 결과는 새로운 LoRA/PCA/metric-learning 알고리즘, SOTA, 독립 일반화 또는 통계적 유의성의 증명이 아니다. 특히 high weak-label agreement를 실제 action 정확도로 쓰면 안 된다. 논문 제목/주제를 미리 고정하지 않고, 독립 주석 결과를 확보한 뒤 주장 범위를 정하는 것이 타당하다.

현재 데이터에서 요청한 학습 모듈의 구현·비교는 완료했다. 다음 [44 계획](EXPERIMENT44_PLAN.md)은 독립 영상과 사람 phase/identity 주석을 먼저 확보하는 검증 단계다. 자료 없이 새로운 독립 평가 결과를 만들지 않았다.
