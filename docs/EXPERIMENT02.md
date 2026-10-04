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

## 결과

| 점수 | AUROC | Average precision |
|---|---:|---:|
| Visual | 0.5787 | 0.3798 |
| Process | 0.5471 | 0.3749 |
| Combined | 0.5959 | 0.4031 |

FIT phase 분포는 `[998, 444, 114]`로 바뀌었다. 하지만 전체 2,893개 sampling 시점 중 직접 bbox 관측은 383개(13.24%)뿐이다. phase 점유율이 분산된 것을 phase 정확도가 검증됐다고 해석하지 않는다.

정상 calibration q99 기준의 테스트 오탐률은 14.85%, 이상 recall은 23.37%다. 실험 01 대비 ranking과 recall은 상승했지만 오탐률도 상승했다. 두 모델의 서로 다른 정상 보정 임계값에서 측정한 값이므로 동일 test FPR에서의 비교가 아니다.

![단계별 성능과 경보 비교](../results/comparison01_02/comparison.png)

[지표](../results/experiment02/metrics.json) · [관측 가용성 및 정상 grounding 근거](../results/experiment02/phase_grounding.json) · [고정 설정](../configs/experiment02.json)

## Recommended improvements

우선순위는 **제품 검출 품질과 관측 누락 처리**다. 특정 색상에 의존한 product prompt와 고정 부품의 역할 혼동을 점검하고, 정상 motion evidence로 product 후보를 검증하는 방식이 후보이다. 주 평가에 사용할 검출 수가 늘었다는 사실만으로 품질을 주장하지 말고, 작은 bbox 검증 subset에서 실제 제품 recall·고정 배경 오검출·누락 지속 길이를 측정해야 한다.

이후 외형/관계/기대 phase 모델 확장은 해당 결과를 보고 결정한다. 실험 03 구현이나 전체 후속 실험 표는 아직 만들지 않았다. 반복 확인한 R01은 개발 장면으로 간주하고, 최종 성능 주장은 별도의 검증 절차와 아직 사용하지 않은 장면에서 확인해야 한다.

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
