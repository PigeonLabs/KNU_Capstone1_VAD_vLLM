# 실험 02 — 정상 trajectory로 phase grounding

## 실험 01의 결과에서 선택한 변경

실험 01의 정상 FIT phase 분포 `[1363, 0, 193]`에서 중앙 상태가 붕괴했다. Process AUROC는 0.4942로 유용하지 않았다. 이에 **phase 추정만** 전체 프레임 CLIP/text similarity에서 정상 trajectory의 공간적 진행 위치로 교체했다.

모델 가중치, detector/vocabulary, crop/full-frame 특징, PCA 규칙, 정상 분할, sampling, 점수 결합, calibration 규칙은 유지한다. 저장된 49개 시퀀스 특징 배열을 그대로 사용하고 `phases` 이외의 모든 배열이 동일한지 assert로 검증한다. PCA·전이 모델·calibration은 새 phase를 사용해 동일한 정상 분할에서 다시 적합한다. 따라서 anomaly score 자체가 동일하다는 뜻은 아니다.

## 정상 데이터로만 적합하는 절차

1. FIT product-role track 중 관측이 3개 이상이고 정규화 좌표의 이동 범위가 0.1 이상인 track을 선택한다.
2. 14개 track의 150개 bbox 중심으로 주 이동축 x를 정한다.
3. 직교축 y의 2–98% 범위에 0.03 여유를 더해 경로 band `[0.3366, 0.4639]`를 얻는다. 하단 고정 부품은 이 band 밖에 있다. **이 필터는 phase anchor 선택에만 사용하며 appearance branch의 박스는 그대로 남는다.**
4. 이동축 위치를 KMeans 3개 군집으로 나누고 좌표 순으로 phase 0/1/2에 대응한다. 중심은 `[0.1087, 0.4191, 0.8255]`다. 이 대응은 R01의 왼쪽/중앙/오른쪽 설명에 한정한다.
5. 추론에서는 현재 sample의 band 내 product bbox를 선택한다. 직전 track을 우선하고, 없으면 confidence가 높은 후보를 선택한다. 최근 유효 위치 최대 3개의 평균으로 가장 가까운 phase를 결정한다.
6. bbox가 없으면 이전 phase를 유지한다. 첫 관측 전에는 phase 0이다. 미래 위치로 보간하지 않는다. 장기간 누락되면 오래된 상태를 유지하는 한계가 있다.

위 거리·관측 수는 실험 02 실행 전에 고정한 heuristic이며 test score로 탐색하지 않았다. spatial phase는 정답 phase가 아니다. 이 실험은 R01에서의 초기 구현 개선이며, arbitrary multi-step grammar에 대한 일반화는 검증하지 않았다.

## 이번 실험 결과

실험 01과 동일한 R01 정상 fit 27개 / calibration 7개, 테스트 15개 영상의 3,685프레임(이상 1,254개), seed 42를 사용했다.

| 점수 | AUROC | Average precision |
|---|---:|---:|
| Visual | 0.5787 | 0.3798 |
| Process | 0.5471 | 0.3749 |
| Combined | 0.5959 | 0.4031 |

FIT phase 분포는 `[998, 444, 114]`로 바뀌었다. 하지만 전체 2,893개 sampling 시점 중 직접 bbox 관측은 383개(13.24%)뿐이다. phase 점유율이 분산된 것을 phase 정확도가 검증됐다고 해석하지 않는다.

정상 calibration q99 기준의 테스트 오탐률은 14.85%, 이상 recall은 23.37%다. 실험 01 대비 ranking과 recall은 상승했지만 오탐률도 상승했다. 두 모델의 서로 다른 정상 보정 임계값에서 측정한 값이므로 동일 test FPR에서의 비교가 아니다.

![단계별 성능과 경보 비교](../results/comparison01_02/comparison.png)

[지표](../results/experiment02/metrics.json) · [관측 가용성 및 정상 grounding 근거](../results/experiment02/phase_grounding.json) · [고정 설정](../configs/experiment02.json)

## 실험 결과의 의의

동일한 검출·encoder 특징에서 phase 추정만 교체하고 하위 정상 모델을 다시 적합했을 때 Combined AUROC는 0.5371→0.5959(차이 +0.0588), AP는 0.3636→0.4031(차이 +0.0395)이 됐다. 이 제한된 비교는 phase 구성 방식이 정상 subspace와 공정 점수의 결과에 영향을 준다는 구현상 근거다. phase 정확도 향상이나 새로운 알고리즘의 novelty를 입증한 것은 아니다.

중앙 FIT 표본이 0개에서 444개로 늘어 초기 phase collapse는 완화됐다. 그러나 정상 오탐률이 6.95%→14.85%로 증가했으므로 경보 품질까지 개선됐다고 결론 내릴 수 없다. 단일 장면·seed의 기술 통계이며 통계적 유의성은 검증하지 않았다.

## 보완할 점

- **직접 관측 부족:** 383/2,893=13.24%는 fit/calibration/test 전체 sampled 시점의 bbox 관측 비율이다. 실제 제품 검출 recall이 아니다. 나머지 시점은 이전 상태 유지 또는 첫 관측 전 초기 phase 0을 사용하므로 오래된 phase가 남을 수 있다.
- **외형 branch의 오검출 잔존:** 경로 band는 phase anchor만 걸러낸다. 고정 부품 등의 crop은 appearance scoring에 남는다.
- **경보 보정의 한계:** 정상 calibration q99를 적용해도 테스트 정상 오탐률은 14.85%다. 원인이 검출, phase 지연, 정상 조건 차이 중 무엇인지 분리하지 못했다.
- **적용 범위와 정답 부족:** R01의 왼쪽→오른쪽 공간 phase에 한정하며 semantic phase 및 객체 localization GT 검증이 없다. R01은 반복 관찰한 개발 장면이므로 최종 일반화 주장은 아직 사용하지 않은 장면과 별도 검증 절차가 필요하다.

## 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 실험 02의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **제품 grounding 품질 개선:** 색상 의존 prompt를 보완하고 정상 이동 경로·motion으로 제품 후보를 검증 | 직접 phase 관측 13.24%, 고정 부품 crop이 외형 branch에 잔존 | 먼저 작은 bbox 검증 subset의 precision/recall·고정 배경 오검출·누락 길이를 측정. 이후 선택한 검출 변경 하나의 효과를 기존 특징/평가 조건과 비교. 정지 제품까지 제거하지 않는지 확인 |
| 2 | **누락 관측을 반영한 phase 불확실성:** 관측 경과 프레임 수, unknown 상태, 재관측 시 상태 갱신을 명시 | 대부분의 시점에서 이전 상태 유지 또는 초기 상태 사용, 오래된 위치가 남을 수 있음 | 검출을 고정한 비교에서 누락 구간 길이·phase age·오탐/recall 측정. 미래 프레임 사용 금지. 제품의 실제 부재를 단순히 무시해 이상 신호를 없애지 않는지 확인 |
| 3 | **신뢰도 기반 score 결합·정상 calibration 개선:** 관측 불확실성에 따라 process 기여를 조절하고 정상 조건별 보정 진단 | Ranking 상승과 함께 정상 오탐률 6.95%→14.85% | 정상 데이터로 결합·임계값 규칙을 결정하고 Visual 단독/고정 결합과 비교. AUROC/AP 및 고정된 calibration 규칙의 오탐/recall 동시 보고. 테스트 임계값 튜닝 금지 |

현재 최우선 후보는 1순위다. 다음 실험을 진행할 때 검출 진단에 근거해 구체적인 변경 하나를 선택한다. 세 후보는 확정된 실험 03·04·05 계획이 아니다. 실험 03은 아직 실행하지 않았으며, 다음 결과에 따라 순위를 다시 정한다.

## 재현

```bash
export PYTHONPATH=src
.venv/bin/python scripts/ground_spatial_phase.py
.venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset \
  --config configs/experiment02.json
.venv/bin/python scripts/plot_experiment.py --experiment 02
.venv/bin/python scripts/compare_experiments.py
```

미래 bbox를 바꾸어도 과거 phase가 변하지 않는 인과성 검사와, 경로 밖 고정 객체가 phase anchor로 선택되지 않는 검사를 포함한다.
