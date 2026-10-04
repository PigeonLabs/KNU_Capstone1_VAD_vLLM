# 실험 06 — 진행량의 phase 조건화 제거

## 질문과 고정 비교

진행량을 phase별로 모델링하는 복잡성이 필요한지 확인했다. 실험 05의 1순위 권고에 따른 ablation이며 [실행 전 계획](EXPERIMENT06_PLAN.md)을 유지했다.

`NormalProgress(condition_on_phase=False)`에서 모든 정상 FIT 진행량을 하나의 median/MAD 모델로 적합한다. 실험 05의 12프레임 진행량·유효 마스크·선택 anchor·detector/ROI·CLIP 특징·phase 배열·정상 분할은 동일하다. 외형 PCA의 phase 조건화와 기존 전이 점수는 유지한다. **전체 파이프라인에서 phase를 제거한 실험이 아니다.**

정상 calibration percentile, max 공정 결합, 0.5/0.5 최종 결합, 정상 q99 정책은 그대로다. 새 residual과 결합 점수에 대해 정상 calibration을 다시 계산했다. 테스트 점수로 설정이나 threshold를 탐색하지 않았다.

## 이번 실험 결과

R01·seed 42, 정상 fit 27개/calibration 7개, 테스트 15개 영상의 3,685프레임(이상 1,254개)이다. 모든 입력 특징과 유효 마스크를 실험 05와 일치시켰다.

| 지표 | 실험 05: phase 조건 | 실험 06: pooled |
|---|---:|---:|
| Visual AUROC | 0.6514 | 0.6514 |
| Visual AP | 0.4260 | 0.4260 |
| Process AUROC | 0.6889 | 0.6929 |
| Process AP | 0.6502 | 0.6502 |
| Combined AUROC | 0.6992 | 0.7049 |
| Combined AP | 0.6584 | 0.6612 |
| 정상 q99 기준 테스트 정상 오탐률 | 5.92% | 7.82% |
| 정상 q99 기준 테스트 이상 recall | 42.26% | 45.45% |
| 첫 경보 지연 중앙값 (원본 프레임) | 38.5 | 12.0 |
| 경보가 발생한 GT 이상 구간 | 8 / 8 | 8 / 8 |
| 진행량 모델 수 (fallback 포함) | 4 | 1 |

각 모델의 정상 calibration q99에서 비교했다. 실험 06 임계값은 0.9792이며 실험 05의 0.9846과 다르다. 동일 테스트 오탐률에서 지연이나 recall을 비교한 것이 아니다. 추가로 계산·저장하던 3개 phase별 진행량 모델을 없애고 pooled 모델 1개만 유지했다. 실제 실행 속도·메모리 절감량은 별도로 측정하지 않았다.

![실험 05–06 비교](../results/comparison05_06/comparison.png)

진행량 유효 비율은 두 실험 모두 sampled 기준 838/927=90.40%, dense 기준 3,350/3,685=90.91%다. 동일한 유효 dense 구간에서 진행량 단독 AUROC는 0.6669→0.6682, AP는 0.6630→0.6618이다. 이 단독 지표는 전체 프레임의 주 지표와 평가 대상이 다르다.

### Phase별 오류와 지연 진단

| phase proxy | 정상 오탐 프레임 05→06 | 이상 탐지 프레임 05→06 |
|---|---:|---:|
| 0 | 0 → 68 | 0 → 88 |
| 1 | 50 → 38 | 430 → 382 |
| 2 | 94 → 84 | 100 → 100 |
| 전체 | 144 → 190 | 530 → 570 |

두 실험의 각 phase에 속하는 정상/이상 프레임 수는 동일하다. phase 0에서는 조기 탐지 가능성과 오탐이 함께 증가했고, phase 1에서는 이상 탐지 프레임이 감소했다. 이 표는 proxy phase의 사후 오류 진단이며 phase GT 검증이나 threshold와 ranking 효과의 분리를 뜻하지 않는다.

GT 양성 구간 8개 중 4개는 첫 경보가 빨라지고 3개는 같았으며 1개는 8프레임 늦어졌다. 대응 구간별 지연 변화 중앙값은 -12프레임이다. 각 실험의 지연 중앙값 차이(38.5→12.0)와 다른 통계다. 미탐은 두 실험 모두 0개이고, 이상 시작 전부터 켜진 경보로 delay 0이 된 사례도 없다. 구간 내 한 번의 경보는 전체 이상 프레임 탐지나 정확한 onset localization을 뜻하지 않는다. 짧은 GT 구간(≤12프레임)은 여전히 0개다.

[전체 지표](../results/experiment06/metrics.json) · [영상별 결과](../results/experiment06/per_sequence.csv) · [phase별 오류](../results/comparison05_06/phase_errors.json) · [이벤트별 지연](../results/comparison05_06/events.json) · [관측 및 source hash](../results/experiment06/progress_features.json)

## 실험 결과의 의의

진행량의 phase 조건을 제거해도 이번 R01에서 Combined AUROC/AP가 낮아지지 않았고, 모델 수는 4→1로 줄었다. 따라서 **진행량에 대한 phase 조건이 반드시 필요하다는 주장을 현재 결과는 지지하지 않는다.** 외형 PCA의 phase 조건이나 언어적 공정 표현까지 불필요하다는 결론은 아니다.

이 결과는 파이프라인의 구성 요소를 늘리는 대신 필요한 복잡성을 분리해 검증한 근거다. pooled 모델을 다음 적용성 실험의 단순한 진행량 모듈 후보로 사용하되, 오탐이 더 낮은 실험 05도 비교 결과로 보존한다. 더 높은 성능만으로 채택하거나 새로운 알고리즘 novelty를 주장하지 않는다.

## 보완할 점

- **상충 관계:** Combined AUROC +0.0057, AP +0.0028의 작은 차이와 함께 정상 오탐률은 5.92%→7.82%로 증가했다. 우월성·통계적 유의성·동등성을 입증한 것은 아니다.
- **보정과 지연의 혼합:** 첫 경보가 빨라졌지만 정상 q99 수치도 달라졌다. pooled 분포가 초기 이상을 더 잘 모델링했다고 단독 귀속할 수 없다.
- **Phase 0 오탐:** 조기 이상 탐지 88프레임 추가와 정상 오탐 68프레임 추가가 함께 나타났다. 정상 진입 구간의 속도 다양성 또는 관측 품질을 더 검증해야 한다.
- **범위 및 독립성:** R01을 반복 개발에 사용했다. 다른 장면·seed·짧은 이상·bbox/phase GT 검증이 없으며, 원본 그룹 독립성도 확인되지 않았다.
- **남은 구조적 제한:** ROI 밖 이상, 실제 제품 부재, 진행량 warm-up 구간은 해결하지 않았다. 시간 단위는 원본 프레임이며 실시간 처리 지연은 미측정이다.

## 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **미사용 실제 장면 적용성 및 공통 실행 구조:** R03 정상 영상으로 객체/공정 discovery와 모델 적합 | 현재 기여·단순화 판단이 반복 관찰한 R01에만 기반 | 테스트 성능과 무관하게 장면 선택. 정상 데이터에서 이동/phase 가정과 검출 가용성을 확인하고, 같은 R03 관측에서 진행량 추가 전후 비교. 적용 불가능한 가정을 강제하거나 실패 장면을 숨기지 않음 |
| 2 | **정상 조건별 보정과 오탐 진단:** 진입/진행/이탈 구간과 관측 신뢰도에 따른 정상 점수 확인 | pooled가 phase 0의 정상 오탐 68프레임을 추가 | 정상 fit/calibration에서만 보정 규칙 정의. R01 test로 phase별 threshold를 최적화하지 않고 오탐/recall/지연을 함께 평가 |
| 3 | **독립 반복과 짧은 이상·누락 검증:** seed/분할 및 이상 지속 길이에 따른 민감도 확인 | AUROC/AP 차이가 작고 짧은 GT 이상이 없음 | 데이터 그룹 독립성 한계를 밝히고 비교 protocol을 먼저 고정. 자연 이상과 인위적 스트레스 검사를 구분하며 영상 단위 불확실성 보고 |

현재 1순위를 [실험 07 계획](EXPERIMENT07_PLAN.md)으로 선택했다. R03은 R01 이후 미사용 실제 장면 중 라벨 길이 정합성이 확보된 장면으로 선정했으며 테스트 성능으로 골랐다는 의미가 아니다. 2·3순위는 확정된 전체 실험 일정이 아니다.

## 재현 및 검증

실험 03의 특징이 필요하다. 저장된 특징을 재사용하며 추가 VLM/GPU 추론은 없다.

```bash
export PYTHONPATH=src
.venv/bin/python scripts/prepare_progress.py --config configs/experiment06.json
.venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset \
  --config configs/experiment06.json
.venv/bin/python scripts/diagnose_motion_errors.py --experiment 06 --baseline 05
.venv/bin/python scripts/compare_event_delays.py --experiments 05 06
.venv/bin/python scripts/diagnose_phase_errors.py --experiments 05 06
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/plot_experiment.py --experiment 06
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/compare_runs.py --experiments 05 06
.venv/bin/python -m pytest -q
```

21개 테스트 통과. pooled 모델의 phase 변경 불변성과 단일 모델 수, 모델 재적합 시 이전 phase 상태 제거를 검증했다. 49개 캐시의 모든 배열이 실험 05와 같고, 15개 영상의 Visual/원래 전이/유효 마스크/anchor/라벨이 같음을 확인했다. 외형 PCA·정상 calibration·전이 모델 배열도 동일하다. [검증 결과](../results/experiment06/validation.json)
