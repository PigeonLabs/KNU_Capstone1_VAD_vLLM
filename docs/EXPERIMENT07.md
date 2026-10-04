# 실험 07 — 미사용 실제 장면 R03 적용성

## 실행 전 선택 및 정상 영상 점검

실험 06의 1순위 권고에 따라 R03을 선택했다. 장면 선정 및 고정 분할은 [계획](EXPERIMENT07_PLAN.md)에 있다. R03은 지게차가 팔레트의 컵을 옮기는 영상으로 R01의 단방향 컨베이어와 다르다.

정상 FIT 01/02의 균등 샘플 12장을 로컬 Qwen에 입력했다. 출력은 `forklift`, `pallet` 역할과 `forklift_idle → forklift_at_pallet → forklift_carrying → pallet_set`의 4개 공정 후보다. 모델이 명시한 실제 lift/release의 추론, camera viewpoint 변화 주장은 독립 검증된 정답이 아니다. 원본 출력과 불확실성은 그대로 보존한다.

## 테스트 평가 전 고정한 비교

- `07`: 전체 프레임 GroundingDINO + tracking + frozen CLIP, CLIP/text phase proxy, phase별 외형 PCA와 전이 점수의 기본 파이프라인.
- `07_motion`: **동일한 R03 검출·특징·phase 배열**에 지게차 bbox 중심의 pooled 진행량을 추가. 지게차는 정상 영상에서 명시적으로 움직이는 기계이며 discovery의 역할 0이다. 역할은 테스트 점수로 선택하지 않는다.
- R01의 product vocabulary, ROI, 왼쪽→오른쪽 spatial phase 지도를 복사하지 않는다. 왕복·정지·lift 동작이 섞여 단조 공간 phase 가정이 맞지 않으므로 이 장면에서는 생성된 언어 phase proxy를 그대로 평가한다.
- 진행량 이동축은 R03 정상 FIT의 충분히 이동한 지게차 track(관측 ≥3, 정규화 이동 범위 ≥0.1)에서 중심 분산이 큰 축으로 적합한다. 현재/이전 ID 우선 anchor를 선택하며 bbox center는 물리적 keypoint가 아니다.
- 4프레임 sampling, 12프레임 진행량, 동일 track의 연속 4개 관측, pooled median/MAD, 유효 residual 정상 calibration, max 공정 결합, 최종 0.5/0.5 가중치와 정상 q99는 실험 06 규칙을 유지한다. 미관측은 기존 전이 점수로 돌아간다.
- signed 진행량은 왕복을 허용하지만 정상 정지·이동을 한 분포에 섞는다. 정지를 이상으로 항상 구분할 수 있다는 가정은 하지 않는다.
- 정상 FIT의 진단 후 두 설정을 고정하고 테스트 평가를 수행한다. R03을 R01과 같은 평가 집합으로 취급하지 않는다. R01에서 적합한 정상 분포를 그대로 전이하는 zero-shot 실험이 아니라, 같은 방법을 R03 정상 데이터에서 새로 적합하는 적용성 실험이다.

## 정상 FIT에서 확인한 적용 범위 (테스트 실행 전)

18개 FIT 영상, 3,142 sampled 시점에서 지게차 anchor 관측률은 100%, 유효 12프레임 진행량은 3,076/3,142=97.90%다. 충분히 이동한 22개 track의 3,101개 관측으로 x축을 적합했다.

정상 진행량은 `|v|≤1e-4`인 near-zero가 40.05%, 양의 이동 30.14%, 음의 이동 29.81%다. 이 기준은 분포 설명용이며 scoring에서 정상 정지를 제거하지 않는다. 정지가 정상에 포함돼 있으므로 R01처럼 정지가 항상 이상이라는 해석은 불가능하다.

FIT phase 분포는 `[2921,48,0,173]`이며 `carrying` 후보의 선택은 0개다. 따라서 전체 공정 grammar가 관측됐다고 주장하지 않는다. 그럼에도 같은 불완전한 phase/외형 관측을 공유한 상태에서 pooled 진행량의 추가 효과를 검증할 수 있으므로 두 설정을 그대로 실행한다.

아래는 고정한 두 설정의 전체 실행 결과다.


## 이번 실험 결과

R03 정상 fit 18개/calibration 4개, 테스트 17개 영상의 12,005프레임 중 이상 5,068개를 평가했다. seed 42다. 39개 영상의 총 6,852 sampled 시점에서 특징을 추출했다. 두 설정은 동일한 특징·phase·평가 라벨을 사용한다.

| R03 지표 | 기본 외형·전이 (07) | 진행량 추가 (07_motion) |
|---|---:|---:|
| Visual AUROC | 0.6968 | 0.6968 |
| Visual AP | 0.6374 | 0.6374 |
| Process AUROC | 0.4980 | 0.5243 |
| Process AP | 0.4223 | 0.4355 |
| Combined AUROC | 0.6747 | 0.6165 |
| Combined AP | 0.5664 | 0.4942 |
| 정상 q99 기준 정상 오탐률 | 3.85% | 3.59% |
| 정상 q99 기준 이상 프레임 recall | 5.56% | 6.41% |
| 경보가 발생한 GT 이상 구간 | 10 / 17 | 11 / 17 |
| 미탐 GT 이상 구간 | 7 | 6 |

각 모델의 정상 calibration q99에서 경보를 측정했다. 임계값은 기본 0.9707, 진행량 추가 0.9770이며 같은 테스트 오탐률 비교가 아니다. 정상 calibration sampled alarm rate는 두 설정 모두 약 1.00%다.

![R03 내부 비교](../results/comparison07_07_motion/comparison.png)

| phase proxy 분포 | idle | at_pallet | carrying | pallet_set |
|---|---:|---:|---:|---:|
| 정상 FIT | 2,921 | 48 | 0 | 173 |
| 정상 calibration | 662 | 1 | 0 | 40 |
| 테스트 | 2,825 | 68 | 0 | 114 |

`carrying`은 정상·테스트 어디에서도 선택되지 않았다. 순서 모델에 네 개의 실질적 공정 상태가 관측됐다는 근거가 없다. 지게차 anchor는 모든 sampled 시점에 존재했고, 진행량은 FIT 97.90%, calibration 97.44%, test 97.71%에서 유효했다. 이 비율은 지게차 localization 정확도나 의미 상태 정확도가 아니다.

진행량 단독 AUROC/AP는 유효 dense 11,729프레임(이상 5,008개)에서 0.5188/0.4536이다. 전체 평가 대상과 다르다. 유효하지 않은 276프레임에서는 기본 설정의 Combined 점수를 그대로 사용했다.

### 이상 구간과 지연

GT 연속 양성 구간 17개 중 공통 탐지 10개, 진행량 추가로 새로 탐지한 구간 1개, 양쪽 미탐 6개다. 기본 설정에서만 탐지한 구간은 없다. 추가 탐지는 R03_05의 `[348,407)` 구간이다.

탐지된 구간만 계산한 첫 경보 지연 중앙값은 기본 241프레임, 진행량 추가 187프레임이다. **탐지된 구간 집합이 다르므로 이 두 중앙값만으로 지연이 개선됐다고 해석하지 않는다.** 공통 탐지 10개 구간의 대응 지연 변화 중앙값은 0프레임이다. 4개 GT 구간은 영상 시작에 걸쳐 있어 왼쪽 경계가 잘렸으며, 12프레임 이하 GT 이상 구간은 0개다. 지연은 원본 프레임 단위이고 FPS는 확인되지 않았다. 미탐은 delay null로 보존했다.

### 관측 품질과 실행 비용

정상 FIT에서 화면 절반보다 큰 bbox의 비율은 지게차 역할 4.12%, 팔레트 역할 6.04%다. 12개 정상 샘플의 정성 점검에서는 팔레트 부분/중복 박스와 바닥 전체 오검출이 보였다. 큰 박스 진단 기준은 이번 scoring에 사용하지 않았으며 모든 큰 박스가 오검출이라는 GT 기반 판정도 아니다.

로컬 Qwen discovery 요청은 39.48초, 39개 영상의 detector/tracker/CLIP/이미지 IO 합계는 632.53초였다. 모델 loading·정상 모델 적합·최종 평가·카메라 대기를 포함하지 않아 end-to-end FPS가 아니다. 추가 유료 API는 사용하지 않았다. 이번 discovery 후 자체 실행한 llama.cpp 서버는 종료했다.

[기본 지표](../results/experiment07/metrics.json) · [진행량 추가 지표](../results/experiment07_motion/metrics.json) · [진행량 및 phase 진단](../results/experiment07_motion/motion_features.json) · [이벤트별 비교](../results/comparison07_07_motion/events.json) · [정상 검출 진단](../results/experiment07/normal_detection_diagnostic.json) · [VLM 응답과 입력](../results/experiment07/process_discovery.json) · [실제 로컬 서버 설정](../results/experiment07/vlm_runtime.json)

## 실험 결과의 의의

R01에서 유용했던 진행량 모듈이 R03에서는 Combined AUROC -0.0582, AP -0.0722의 저하를 보였다. 진행량을 대부분의 시점에서 계산할 수 있어도 공정 이상에 유용한 신호가 된다는 보장은 없다. 실제로 R03 정상 영상에는 정지·양방향 이동·적재 관계 변화가 섞여 있어, 하나의 지게차 중심 속도만으로 공정 상태를 설명하기 어렵다. 이 설명은 관측에 부합하는 가설이며 모든 오류의 인과적 원인을 입증한 것은 아니다.

파이프라인 구현 측면에서는 장면별 vocabulary/discovery/캐시/평가 경로와 역할 기반 motion anchor를 분리하고, R01 전용 공간 가정을 다른 공정에 억지로 적용하지 않는 구조를 확보했다. 논문에는 **모듈의 적용 조건과 실패 사례**를 설명하는 근거로 사용할 수 있다. R01 수치를 다른 장면에서 재현했다거나 알고리즘 novelty·일반화 우월성을 입증했다고 주장하지 않는다.

## 보완할 점

- **공정 상태 관측 실패:** carrying phase가 모든 분할에서 0개다. 텍스트로 공정 후보를 생성하는 단계와 실제 영상에 grounding하는 단계의 간극이 남았다. 기본 Process AUROC도 0.4980이다.
- **관계 정보 부족:** 지게차의 이동량은 팔레트가 들어올려졌는지, 함께 움직이는지, 내려놓였는지를 직접 표현하지 않는다. 정상 정지도 한 pooled 분포에 포함돼 있다.
- **검출 불확실성:** 높은 역할 관측률에 큰 배경 박스·부분 팔레트·중복 박스가 섞인다. 독립 bbox/phase GT 기반 정확도 검증이 없다.
- **경보 품질:** 두 설정 모두 이상 프레임 recall이 6% 안팎이며 많은 구간을 놓친다. 경보가 있는 경우에도 늦을 수 있다. Visual 단독 AUROC/AP 0.6968/0.6374가 두 결합 점수보다 높다.
- **평가 범위:** 한 R03 분할/seed의 적용성 결과다. 이번 결과로 다음 개선을 선택하면 R03도 개발 장면으로 취급한다. 짧은 이상·그룹 독립성·최종 외부 검증·유의성은 아직 확보되지 않았다.

## 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **객체 관계 기반 상태 grounding:** 지게차–팔레트 상대 위치·겹침·크기 관계로 정상 latent state를 구성 | carrying 상태 미관측, 지게차 단독 진행량 추가로 ranking 저하 | 정상 데이터에서 관계 관측의 품질·상태 support를 먼저 확인하고 phase 모듈을 비교. cluster 점유율을 semantic 정확도로 부르지 않으며 테스트 시간/미래 프레임을 입력으로 사용하지 않음 |
| 2 | **역할별 검출 품질과 관측 불확실성:** 큰 배경 박스, 부분/중복 팔레트 후보를 진단·보완 | 정상 FIT 큰 박스 비율과 정성 오검출 관찰 | 정상 bbox 검증 subset 또는 명시적인 정성 검토로 선택 규칙을 점검. GT 없는 역할 coverage를 recall로 보고하지 않고 실제 부재와 detector 실패를 구분 |
| 3 | **공정 모듈의 조건부 사용과 결합 검증:** Visual 단독 및 관측 신뢰도 기반 공정 기여 비교 | 기본/진행량 Process가 약하고 결합이 Visual 단독보다 낮음 | 정상 support·예측 일관성으로 규칙을 고정하고 test 지표로 가중치를 탐색하지 않음. 상태별 오류·오탐/recall·미탐/지연을 함께 보고 |

1순위를 [실험 08 계획](EXPERIMENT08_PLAN.md)으로 선택한다. 이번 진행량 모듈의 저하를 감추거나 R03에서 다시 가중치를 탐색하지 않는다. 이후 전체 실험 일정은 고정하지 않는다.

## 재현 및 검증

저장된 discovery를 재사용하면 서버 실행과 discovery 재생성을 생략한다. Qwen은 temperature 0.2와 기본 seed를 사용하므로 재생성 응답이 동일하다고 보장하지 않는다. 원본 데이터·가중치·특징·검토 이미지는 GitHub에 업로드하지 않는다.

```bash
export PYTHONPATH=src
# 별도 터미널에서 bash scripts/start_local_vlm.sh
.venv/bin/python scripts/discover_process.py --data-root /path/to/IPAD_dataset --config configs/experiment07.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/extract_features.py \
  --data-root /path/to/IPAD_dataset --config configs/experiment07.json --fit-only
.venv/bin/python scripts/prepare_scene_motion.py --config configs/experiment07_motion.json --fit-only
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/extract_features.py \
  --data-root /path/to/IPAD_dataset --config configs/experiment07.json
.venv/bin/python scripts/prepare_scene_motion.py --config configs/experiment07_motion.json
.venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset --config configs/experiment07.json
.venv/bin/python scripts/evaluate_baseline.py --data-root /path/to/IPAD_dataset --config configs/experiment07_motion.json
.venv/bin/python scripts/diagnose_motion_errors.py --experiment 07_motion --baseline 07
.venv/bin/python scripts/compare_event_delays.py --experiments 07 07_motion
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/plot_experiment.py --experiment 07
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/plot_experiment.py --experiment 07_motion
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/compare_runs.py --experiments 07 07_motion
.venv/bin/python -m pytest -q
```

25개 테스트 통과. 다른 장면 discovery의 암묵적 재사용 거부, 정상 source/phase ID 검사, R01 corridor 없는 역할 anchor와 정상 이동축 적합을 검증했다. 39개 영상에서 모든 기본 특징 배열이 보존되고 17개 테스트에서 Visual·원래 전이·라벨·미관측 fallback이 동일함을 검사했다. 모델·설정의 테스트 평가 전 고정 hash도 일치한다. [검증 기록](../results/experiment07/validation.json) · [실행 전 설정 고정](../results/experiment07/pre_evaluation_protocol.json)
