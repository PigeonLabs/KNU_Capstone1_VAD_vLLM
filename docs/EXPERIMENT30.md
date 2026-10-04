# 실험 30 — 객체 쌍 track 경계에서 관계 평균 초기화

상태: 완료. R04 정상 FIT/calibration 준비, 30개 정상 holdout cell, 6개 구성×19개 테스트 평가 및 저장 결과 재구성을 마쳤다. [사전 계획](EXPERIMENT30_PLAN.md)은 실행 당시 미실행 상태의 동결 문서로 유지한다.

## 1. 이번 실험 결과

### 질문·변경·고정 조건

선택된 객체 쌍이 바뀌었는데 이전 객체의 관계 descriptor가 평균에 남는 현상을 제거하면 전체 파이프라인이 어떻게 달라지는가? 실험 29에서 발견한 FIT 114/calibration 26개의 mixed-pair window가 출발점이다.

`reset_on_track_change=True`일 때 선택된 anchor 또는 target track ID가 바뀌면 길이 3 관계 평균을 초기화한다. 동일한 연속 관측 쌍에서는 기존 평균을 유지한다. 누락 시 이력 초기화와 마지막 phase 유지, 최초 phase 0 규칙도 유지한다. semantic margin의 track별 median은 이번 변경 대상이 아니다.

실험 18의 relation 중심/location/scale/area gate와 객체 선택, 임베딩, 관측 mask를 고정했다. 새 phase로 정상 appearance PCA·전이 확률·체류 분포·CDF/q99를 다시 적합했다. 실험 27의 full-transition CDF와 consecutive-observed gate를 사용한다. 실험 28의 observed-only CDF는 채택하지 않았다.

외형 경로는 hold(누락 시 마지막 phase), pool(누락 즉시 pooled), age(정상 FIT에서 동결된 τ=56프레임 이내 마지막 phase, 이후 pooled) 세 가지다. 어느 경로도 테스트 성능으로 선택하지 않았다. 공유 phase bank rank는 두 자연 rank와 실험 27 상한의 최솟값이다. 실제 공통 map은 실험 27과 같았고 양쪽 모두 phase 1/2/3의 12개 bank를 지원한다. global phase 2는 30, role 1 phase 2는 29, 나머지는 32다. phase 0은 양쪽 모두 미지원, pooled bank 4개의 평균/기저는 완전히 같다. control의 정상 모델과 테스트 prediction 57개는 실험 27 저장본과 배열 단위로 일치했다.

### 데이터·실행 범위

| 항목 | 범위 |
|---|---|
| 정상 FIT | R04 20개 영상, 7,812프레임 / 1,960 samples |
| 정상 calibration | 별도 5개 영상, 1,920프레임 / 482 samples |
| 정상 holdout | calibration 영상 하나를 CDF/q99에서 제외, FIT 고정; 6구성×5회 |
| 테스트 | 19개 영상, 8,154프레임 / 2,047 samples |
| 라벨 분포 | 정상 3,576 / 이상 4,578프레임, 연속 GT 구간 26개 |
| split / seed / 시간 | 기존 sequence split, seed 42, stride 4, source frame index |
| 추론·평가 | normal-only FIT/calibration, causal zero-order hold, strict score > own q99, point adjustment 없음 |
| 모델 호출 | 캐시 재사용; VLM/검출기/CLIP 재호출 없음. 실행 비용·지연은 미측정 |

R04에서는 라벨 길이가 모두 일치한다. 전체 데이터셋의 11개 길이 불일치 시퀀스는 여전히 정렬 미확정이고 원본을 수정하지 않았다. FPS·timestamp·semantic phase/object GT·독립 녹화 그룹은 없다. 이 테스트는 반복 사용한 개발 평가다.

### 탐지 결과

| 구성 | Visual AUROC / AP | Combined AUROC / AP | 정상 오탐률 (FP) | 이상 recall (TP) | 탐지 / 26 |
|---|---:|---:|---:|---:|---:|
| control_hold | 0.6899 / 0.6899 | 0.6820 / 0.6816 | 8.19% (293) | 13.65% (625) | 13 |
| control_pool | 0.6875 / 0.6857 | 0.6797 / 0.6774 | 7.41% (265) | 11.97% (548) | 14 |
| control_age | 0.6898 / 0.6891 | 0.6820 / 0.6808 | 8.19% (293) | 13.83% (633) | 13 |
| reset_hold | 0.6905 / 0.6896 | 0.7158 / 0.7083 | 9.09% (325) | 18.20% (833) | 16 |
| reset_pool | 0.6889 / 0.6861 | 0.7154 / 0.7064 | 8.08% (289) | 16.78% (768) | 17 |
| reset_age | 0.6902 / 0.6886 | 0.7156 / 0.7074 | 9.09% (325) | 18.37% (841) | 16 |

![실험 30 비교](../results/experiment30/track_reset_comparison.png)

hold/pool은 짝지은 control과 q99가 같다. age는 0.9972375691→0.9974576271이다. 모든 구성의 calibration 경보는 4/482 samples, 점수 1은 0개이며 finite/[0,1]/q99<1을 통과했다. 정상 holdout 30개 cell도 모두 통과했다.

| 경로 | control → reset q99 | 정상 holdout FP / 1,920 | 조건부 지연 중앙값, source frames | 변화 FP / TP |
|---|---:|---:|---:|---:|
| hold | 0.9974576271 → 동일 | 32 → 32 | 75 → 70.5 | +32 / +208 |
| pool | 0.9968879668 → 동일 | 32 → 40 | 64.5 → 65 | +24 / +220 |
| age | 0.9972375691 → 0.9974576271 | 32 → 32 | 75 → 70.5 | +32 / +208 |

지연 중앙값은 **탐지된 구간만** 대상으로 하며 새 구간이 추가되어 모집단도 다르다. 미탐은 별도로 13→10/12→9/13→10개다. 모든 구성에서 시작 전부터 경보가 켜진 탐지 구간은 2개다. GT 길이 ≤12프레임인 짧은 구간은 없다. 새 탐지 구간은 테스트 08(start 134), 11(start 170), 19(start 166)이며 잃은 구간은 없다.

동일 reset-age 점수에 control-age q99를 적용한 사후 진단은 FP 325/TP 849다. 자체 q99에서는 TP 841로 8프레임 줄지만 탐지 구간은 16개로 같다. hold/pool은 임계값 효과가 없다. 이 비교는 threshold 효과 분리용이며 test로 임계값을 선택하지 않는다.

### 관계 평균·phase·지원 변화

| 파티션 | 유효 관계 samples (불변) | mixed window control → reset | phase가 바뀐 samples / dense frames |
|---|---:|---:|---:|
| FIT | 1,123 | 114 → 0 | 77 / 308 |
| calibration | 297 | 26 → 0 | 13 / 52 |
| test | 1,406 | 120 → 0 | 53 / 212 |

선택 detection index, valid mask, raw geometry, bbox/track/confidence 및 CLIP 특징은 보존된다. 유효 평균에서 여러 track 쌍이 섞이지 않음을 raw descriptor history로 독립 재구성했다. 관측률은 검출 recall 또는 의미 정확도가 아니다. 테스트 연속 관측 전이 가용성은 양쪽 4,975/8,154프레임으로 같다.

FIT 관측 phase 표본은 `[3,698,94,328]`→`[1,712,93,317]`, calibration은 `[0,146,31,120]`→`[0,151,30,116]`이다. 자연 PCA rank도 global phase 2는 30→31, role 1 phase 2는 31→32였지만 공통 rank 30/29로 제어했다. 상세 표본 수·모든 bank는 [rank audit](../results/experiment30/fit_rank_control.json)에 있다.

| 체류 맥락 | control 완결 runs / 영상 수 | reset 완결 runs / 영상 수 | 지원 변화 |
|---|---:|---:|---|
| 1→3 | 12 / 9 | 14 / 10 | 양쪽 지원 |
| 3→1 | 8 / 5 | 10 / 6 | 미지원 → 지원 (최소 10 runs) |

정상 FIT 전이 빈도와 체류 분포는 새 phase로 재적합되어 변했다. 기존 중심은 그대로지만 표본·support·PCA·공정 재적합 효과가 포함된 **전체 파이프라인 비교**다. 평균 초기화의 추론 시점 순수 효과로 해석할 수 없다.

### 공정·체류의 독자 기여

공정 단독 AUROC/AP는 control 0.5679/0.5981→reset 0.5946/0.6615다. 체류 가용성은 417→2,433/8,154프레임(5.11%→29.84%). reset의 기존 1→3은 433프레임, 새 지원 3→1은 2,000프레임이다. 3→1의 정상 FIT 길이는 4~60프레임, 중앙값 10이며 10개 run/6개 영상뿐이다.

아래는 각 구성의 **Combined 정상 q99**에서 외형·다른 공정 branch가 경보하지 않는 독자 경보다. 별도 Visual q99를 사용하지 않았다.

| 경로 | control 전이 독자 FP / TP | reset 전이 독자 FP / TP | reset 체류 독자 FP / TP | reset 공정 추가 탐지 구간 |
|---|---:|---:|---:|---:|
| hold | 16 / 4 | 16 / 4 | 12 / 180 | 1 |
| pool | 16 / 4 | 16 / 4 | 12 / 196 | 1 |
| age | 16 / 4 | 16 / 4 | 12 / 180 | 1 |

control 체류 독자 경보는 0/0이다. reset 체류 독자 경보는 전부 새 지원 3→1에서 발생하며, 외형만으로 못 잡던 테스트 08의 start 134 구간을 추가했다. 나머지 새 탐지 두 구간은 새 외형 점수에서도 탐지한다. 체류 독자 FP 12개는 같은 영상의 정상 라벨 구간 `[244,256)`에 있고 시작 age=172프레임이다. 정상 FIT 범위를 넘는 꼬리 점수가 이상만 가리킨다는 보장은 없다. 가용성 증가나 새 탐지만으로 의미 있는 공정 이해가 검증된 것은 아니다.

### 정상 반례와 새 오류

calibration 02의 frame 152/376은 모두 평균 phase 3→2에서 reset 2→2로 바뀌었다. 기존 29 진단과 부합하지만 376 앞의 3→2 전이는 frame 372로 이동한다. 해당 전이 유형 전체가 사라진 것은 아니다. 이 두 지점은 실험 28에서 오탐이었고, 이번 control은 실험 27 CDF여서 이미 q99 미만이다. 따라서 두 지점의 점수 하락을 이번 실험의 오탐 제거로 계산하지 않는다.

새 정상 오탐은 02의 frame 76/252에서 나타났다. 둘 다 관측 2→1 전이이고 anchor track이 `[7,4]→[1,4]`, `[18,9]→[19,9]`로 바뀐다. 다른 4개 calibration 영상에는 이 edge가 없고, 이전 상태 reference 31개가 있어도 raw=4.17484가 reference max=2.90658을 넘어 percentile 1이 된다. 두 경계는 총 8프레임이다. hold/age에서는 일부 외형 오탐 감소와 상쇄되어 총 정상 FP가 같아졌고, pool에서는 8프레임 늘었다. 같은 정상 FP 합계가 같은 오류 사례를 의미하지 않는다.

## 2. 실험 결과의 의의

객체 관계의 평균이 서로 다른 선택 track을 섞던 구현상 문제를 명시적인 경계 규칙과 검증으로 제거했다. 동일 raw 특징/선택/관측 및 고정 phase 좌표계에서 파생 상태와 후속 정상 모델이 어떻게 바뀌는지 추적할 수 있다. control의 기존 결과 재현과 공통 rank 대조는 비교의 신뢰도를 높인다.

이번에는 결합 점수가 Visual보다 높은 ranking을 보이고 체류의 독자 구간 탐지도 관측됐다. 다만 새 체류 support가 8→10 runs로 임계 경계를 넘은 것이 함께 작용했다. novelty 후보는 성능 수치 자체보다 **객체 identity 경계를 고려한 관계 표현과 공정 근거 추적을 결합한 파이프라인 구현**에 있다. 기존 연구 대비 신규성, 의미 정확도, 통계적 유의성, 다른 장면 일반화는 아직 입증하지 않았다. 논문 주제를 지금 고정하지 않는다.

## 3. 보완할 점

- 모든 경로에서 테스트 FP가 증가했고 정상 pool holdout도 악화됐다. 평균 혼합을 없애도 다른 객체의 상태를 이어 비교하는 전이 점수에는 문제가 남는다.
- track ID는 실제 identity 정답이 아니다. track 단절이 평균을 불필요하게 초기화할 수 있고, 잘못된 역할을 계속 같은 track으로 잡는 경우도 해결하지 못한다.
- phase 중심은 기존 평균 분포에서 학습되어 reset 분포와 맞지 않을 수 있다. 중심 재학습은 이번에 섞지 않았다.
- 새 3→1 체류 모델은 지원 경계에 걸친 10 runs/6영상이다. support 증가와 파라미터 재적합의 기여가 분리되지 않았고 normal calibration 5영상의 전이 커버리지도 부족하다.
- 동일 rank는 동일 표본 수·동일 기저·동일 분포를 뜻하지 않는다. phase/PCA/전이/체류/CDF가 함께 변한다. 학습 데이터 없이 test phase만 교체한 비교가 아니다.
- R04 단일 split/seed의 반복 개발 결과이며 프레임은 독립 표본이 아니다. 다른 장면, 추가 독립 정상 영상, 검증된 객체/행동 GT가 필요하다. localization 정확도·운용 지연·비용은 미측정이다.

## 4. 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **선택된 track 쌍의 연속성으로 전이 증거 검증**: 두 sample 모두 관계가 관측되고 선택된 anchor/target ID가 모두 같을 때만 전이 점수를 결합 | 새 정상 2→1 오탐 2곳 모두 anchor track 교체 경계. 나머지 정상 CDF의 상태 support 31개에도 해당 edge는 0개이며 점수 1. 전이 독자 기여는 여전히 FP 16/TP 4 | 평균·phase·정상 FIT 확률/CDF·외형·체류는 고정하고 전이 결합 gate만 대조. 정상 holdout과 독자 FP/TP, 놓치는 구간 및 q99 효과 검사. 실제 정상/이상 객체 교체나 track 단절의 유효 전이까지 제거할 수 있음 |
| 2 | **체류 맥락 지원을 맞춘 기여 분리**: 공통 지원 1→3과 새 지원 3→1의 사용 여부를 명시적으로 나누는 대조 | 3→1 완결 run이 8→10개로 최소 지원 경계를 넘어 가용 프레임 2,000개가 추가됐고 체류 독자 경보 전부가 이 맥락에서 발생 | 정상 FIT만으로 지원 집합을 동결하고 각 구성의 정상 q99·holdout 재검증. 새 맥락은 6개 영상/10 runs뿐이며 꼬리 분포·역할 정확도를 보장하지 않음. 지원 효과를 초기화 자체의 순수 효과로 부르지 않음 |
| 3 | **정상 시공간 근거를 이용한 객체 역할 검증**: blade/바이스 및 target 혼동 사례를 정상 데이터에서 검토하고 검증 신호를 보강 | 평균 혼합은 0이지만 선택 객체·semantic margin은 전혀 바뀌지 않았고 실험 29의 역할 혼동이 남음. 테스트 FP는 24~32프레임 증가 | 검토용 주석과 normal-only 학습을 구분하고 선택 정확도·관측률·최종 탐지를 따로 검사. 목적 표집 메모를 의미 GT로 취급하거나 test로 cutoff를 조정하지 않음 |

1순위만 [실험 31 계획](EXPERIMENT31_PLAN.md)으로 구체화했다. 다른 두 후보를 확정된 후속 실험으로 취급하지 않는다. 관측된 탐지 개선과 오탐 증가를 모두 유지한 채 다음 변경의 역할을 검증한다.

## 검증·산출물·재현

- 단위 테스트 107개 통과. 동일 쌍 불변, anchor/target 교체, 누락, 시작, causal prefix 포함.
- 정상 holdout 30 cell, 테스트 prediction 114개, 파생 feature 88개를 재구성했다. 정상 reference/CDF/q99와 공정 gate를 독립 계산으로 확인했다. control 저장본 57개와 실험 27 모델/점수도 완전 일치했다.
- 최초 준비 검증에서 raw geometry의 float32 평균을 검증기가 float64로 승격하여 최대 1.19e-7 차이로 중단했다. PCA/calibration/test 전 검증기 dtype만 고쳤다. 생산 알고리즘/설정은 바꾸지 않았으며 [최초 protocol](../results/experiment30/pre_normal_attempt01.json), [수정 기록](../results/experiment30/preparation_repair.json), [최종 정상 사전 동결](../results/experiment30/pre_normal_protocol.json)을 보존한다. 선택적 실패 은폐나 결과 기반 재설계가 아니다.
- [FIT/rank 동결](../results/experiment30/pre_calibration_checkpoint.json) → [정상 feasibility](../results/experiment30/normal_audit.json) → [test 전 동결](../results/experiment30/pre_test_checkpoint.json) 순서로 진행했다. [mechanism audit](../results/experiment30/mechanism_audit.json)은 결과 이후 설명용 진단이며 점수/파라미터를 변경하지 않았다.
- [비교 CSV](../results/experiment30/comparison.csv), [전체 진단·구간·영상별 결과](../results/experiment30/diagnostic.json), [검증](../results/experiment30/validation.json), [normal transform](../results/experiment30/normal_transform.json), [test transform](../results/experiment30/test_transform.json).
- 설정은 `configs/experiment30_{control,reset}_{hold,pool,age}.json`, 각 구성 상세 지표와 영상별 CSV는 `results/experiment30_*/`에 있다. 특징·모델·예측 배열은 ignored `artifacts/`에만 저장하고 원본 영상/가중치/캐시/로그는 업로드하지 않는다.

기존 실험 19 feature cache, 실험 18 relation 모델 및 실험 17 text embedding이 필요하다. 역사적 동일성 audit에는 실험 27 저장본도 필요하다. hash checkpoint는 해당 실행을 기록하므로 소스/라이브러리를 바꾼 재현은 별도의 실행 디렉터리에서 수행한다.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/experiment30_track_reset.py prepare
PYTHONPATH=src .venv/bin/python scripts/experiment30_track_reset.py normal
PYTHONPATH=src .venv/bin/python scripts/experiment30_track_reset.py test_prepare
# normal_audit.json의 eligible에 포함된 구성만 평가
for group in control reset; do
  for gate in hold pool age; do
    PYTHONPATH=src .venv/bin/python scripts/evaluate_baseline.py \
      --data-root /media/jeong/ExtHDD/IPAD_vLLM/IPAD_dataset \
      --config "configs/experiment30_${group}_${gate}.json"
  done
done
PYTHONPATH=src .venv/bin/python scripts/diagnose_track_reset.py
PYTHONPATH=src .venv/bin/python scripts/audit_track_reset_mechanism.py
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python scripts/plot_track_reset.py
```
