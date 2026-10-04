# 실험 29 — 정상 전이 커버리지와 관측 근거 추적

## 이번 실험 결과

**같은 공정 역할로 선택된 객체가 바뀌어도 관계 평균이 이전 track의 특징을 유지하는 현상을 확인했다.** 실험 28의 두 정상 오탐 반례에서는 바이스 주변 anchor가 올라간 blade 주변으로 바뀐 뒤, 서로 다른 track의 descriptor가 섞여 잠재 phase 3→2가 만들어졌다. 정상 전이 커버리지 부족과 관측/평균 처리 문제가 함께 존재한다.

이번은 정상-only 진단 기능 구현이다. 기존 특징·phase·점수·모델·임계값은 수정하지 않았다. **새 AUROC/AP·FPR·recall·탐지 지연은 평가하지 않았다.** 테스트 영상/라벨을 새로 열지 않았고 실험 28의 수치를 이번 성능으로 재사용하지 않는다.

### 구현 범위

정상 FIT 20 + calibration 5개, 원본 9,732프레임과 4프레임 간격의 2,442 samples를 사용했다. 영상 경계를 제외한 전이는 2,417개다. 실험 18에서 고정한 관계 모델과 실험 19 특징 캐시를 사용해 다음을 연결했다.

- 전이 ID → 이전/현재 source frame·phase·관계 mask.
- 선택된 anchor/target의 검출 인덱스·role·track·bbox·confidence와 모든 후보.
- anchor의 raw/temporal semantic margin, raw 관계 descriptor, 3-sample 평균, 기존 중심까지의 거리.
- 영상별 전이 count와 고유 영상 수, calibration holdout의 사라진 전이.

phase/mask·선택 객체·descriptor·margin을 저장 모델에서 정확히 재구성했다. 없는 target semantic margin은 null로 남겼다. 정상의 모든 sampled source path와 선택 bbox/track을 검증했다. local trace와 원본 contact sheet는 `artifacts/experiment29/`에만 저장하고 GitHub에는 코드·집계·사례 ID/좌표·메모만 올린다.

### 정상 전이의 커버리지

| 항목 | FIT 20개 영상 | Calibration 5개 영상 |
|---|---:|---:|
| Sampled frames | 1,960 | 482 |
| 전체 인접 전이 | 1,940 | 477 |
| 연속 관측 전이 | 964 | 247 |
| 전체 3→2 전이 / 영상 수 | 11 / 9 | 6 / 3 |
| 연속 관측 3→2 전이 / 영상 수 | 4 / 4 | 2 / 1 |

![관측 전이 커버리지](../results/experiment29/observed_transition_coverage.png)

관측 3→2는 FIT 영상 01/03/04/17에서 각 1회, calibration 영상 02에서 2회였다. calibration holdout에서 제외 영상의 관측 전이가 다른 영상에 전혀 없는 경우는 영상 02의 3→2 두 곳뿐이다. 이때 이전 상태 3 표본은 여전히 88개다. count≥10인 상태 support가 특정 전이의 커버리지를 보장하지 않는다.

이 수는 잠재 phase ID의 전이이며 arrival/grasp 등의 의미 공정 순서가 아니다. 자기 유지 전이가 많고 영상 내 표본은 상관되므로 count를 독립 표본 수로 해석하지 않는다.

### 정상 반례를 source frame까지 추적

| 사례 | 평균의 source frames | 평균의 anchor track IDs | 저장 phase | 현재 raw descriptor의 최근접 기존 중심 |
|---|---|---|---:|---:|
| 영상 02, frame 148 | 140, 144, 148 | 11, 11, 13 | 3 | 2 |
| 영상 02, frame 152 | 144, 148, 152 | 11, 13, 13 | 2 | 2 |
| 영상 02, frame 372 | 364, 368, 372 | 26, 26, 27 | 3 | 2 |
| 영상 02, frame 376 | 368, 372, 376 | 26, 27, 27 | 2 | 2 |

두 반례에서 target track은 9로 유지됐다. current/previous sample만 보면 anchor track도 각각 13→13, 27→27이지만 평균 이력에는 앞선 11/26이 남아 있다. 시각 검토상 앞선 bbox는 우하단 바이스 주변, 새 bbox는 올라간 금속 blade 주변이다. 따라서 단순히 직전 두 sample의 track 동일 여부만 검사하면 이 혼합을 놓친다.

최근접 중심 값은 **저장된 raw descriptor와 기존 중심의 거리 진단**이다. 새 phase를 적용하거나 모델·점수를 재평가한 결과가 아니다. 평균 처리에 의해 이 두 잠재 전이가 만들어지는 수치적 경로는 확인했지만 실제 의미 공정 단계가 바뀌었는지의 정답은 없다.

### 객체 쌍이 섞이는 평균 범위

관계 평균은 현재 구현에서 누락 시에만 초기화되고 선택 track 변경 시에는 유지된다.

| 정상 subset | 유효 관계 samples | 서로 다른 쌍이 섞인 평균 | anchor 혼합 | target 혼합 |
|---|---:|---:|---:|---:|
| FIT | 1,123 | 114 | 34 | 81 |
| Calibration | 297 | 26 | 13 | 14 |

anchor/target 혼합은 중복 가능하므로 단순 합산하지 않는다. 전체 정상 관측 3→2 6회 모두 현재 평균에 서로 다른 쌍이 섞여 있었다. FIT 4회 중 3회는 그 전이 자체에서 anchor track이 바뀌었고, calibration 2회와 FIT 영상 17은 더 이른 변경의 이력을 유지한 경우였다. 이 연관을 모든 혼합이 오류라는 판정이나 인과적 성능 향상으로 확대하지 않는다.

### 30개 사례의 시각 검토

집계를 마친 뒤 원본 이미지를 보기 전에 선정 규칙과 사례 ID/hash를 저장했다. 관측 3→2가 총 6개여서 모두 포함했고, 이전 상태 3의 다른 목적 상태(자기 유지 포함)를 영상별 round-robin으로 24개 골랐다. 총 30개 사례·15개 local contact sheet를 검토했다. 각 sheet는 이전/현재/다음 sample을 보여주며 다음 sample은 사후 설명용이다. 원인 확인을 위해 영상 02의 frame 144/368 원본도 추가로 확인하고 별도 hash를 남겼다.

- FIT 영상 01/03/04에서는 anchor가 우하단 바이스 주변에서 다른 금속 부위로 이동했다. 영상 01은 작은 blade 부착부 주변 bbox로, 정확한 역할 대표성은 미확정이다.
- 대조 사례에서도 바이스가 anchor로 선택되는 현상이 반복됐다. 양수 semantic margin과 관계 valid가 올바른 역할의 증거는 아니다.
- FIT 영상 20의 대조 사례는 anchor가 바이스→blade→바이스로 바뀌는데 저장 phase는 3으로 유지됐다.
- FIT 영상 23/25 등에서는 target bbox가 갈색 제품 대신 올라간 금속 blade 주변을 선택하는 것으로 보였다. 다른 작은 bbox는 제품 여부를 확정하지 않았다.

이 기록은 **assistant의 탐색적 시각 관찰**이며 독립 인력의 semantic/bbox 주석이 아니다. 256×256 이미지, 목적 표집, 정상 영상에 한정돼 semantic accuracy/검출 recall을 산출하지 않았다. [사례별 검토 메모](../results/experiment29/visual_review.json)

## 결과의 의의

전이 이상 점수를 원본 관측까지 추적하는 진단 기능을 추가해, 단순 보정 수치 조정으로는 드러나지 않던 객체 선택·평균 이력의 상호작용을 확인했다. 정상 커버리지 부족뿐 아니라 여러 객체의 관계 특징을 하나의 시간 이력으로 평균하는 구조가 잠재 전이에 영향을 줄 수 있음을 구체적인 정상 반례로 보여준다.

이는 다음 파이프라인 변경의 근거다. 이번에 탐지 성능을 높였다는 주장은 없고, 이 구현만으로 novelty·통계적 유의성·다른 장면 일반화를 주장하지 않는다. 논문 주제는 아직 좁혀 고정하지 않는다.

## 보완할 점

- track ID는 실제 객체 identity 정답이 아니다. 같은 객체가 track 단절로 새 ID를 받을 수 있어 무조건적인 이력 초기화는 노이즈를 늘릴 수 있다.
- 평균을 초기화해도 바이스를 anchor로 고르는 지속적 역할 오류는 해결되지 않는다. target의 의미도 독립적으로 검증되지 않았다.
- 현재 phase 중심 자체도 기존 평균으로 학습됐다. 고정된 중심에서 평균 규칙을 바꿀 경우 분포 이동을 확인해야 한다.
- 시각 검토는 30개 목적 표집 사례뿐이며 전체 정상/테스트의 의미 정확도를 대변하지 않는다. FPS·독립 녹화 그룹·semantic/bbox GT가 없다. 라벨 불일치 11개는 정렬 미확정 상태로 남는다.
- 정상-only 감사로 test 탐지 효과와 운용 지연/비용은 평가하지 않았다.

## 다음 Recommended improvements — 추천순 3개

| 추천순 | 개선 후보와 변경 내용 | 이번 결과의 근거 | 검증 기준 및 주의점 |
|---|---|---|---|
| 1 | **객체 쌍 track 경계에서 관계 평균 초기화**: anchor 또는 target track이 바뀌면 이전 descriptor 이력을 비움 | 정상 3→2 6회 모두 현재 평균에 서로 다른 쌍이 섞임. 두 calibration 반례의 raw 최근접 중심은 2→2인데 저장 평균 phase는 3→2 | 선택 객체/관측 mask·기존 phase 중심을 고정하고 평균 규칙만 변경. 정상 재적합·holdout 후 테스트, rank/support 변화도 보고. track 단절이 실제 객체 교체라는 보장은 없음 |
| 2 | **정상 시공간 근거를 추가한 역할 검증**: blade와 바이스·갈색 제품 후보의 혼동을 줄이는 검증 | 검토 사례에서 양수 semantic margin으로 바이스를 anchor로 선택하는 현상과 target의 blade 선택이 보임 | 정상 데이터에서만 규칙을 정하고 역할 검토·가용성·최종 탐지를 함께 확인. 이번 목적 표집을 정확도 근거로 쓰거나 test로 margin을 조정하지 않음 |
| 3 | **고정 설정의 다른 장면 적용성 검증** | 전이 3→2의 calibration support가 한 영상에 집중하며 현재 관측 문제도 R04 사례에 한정 | 장면/protocol을 먼저 고정하고 정상 적합 뒤 test 한 번 평가. 다른 기구/객체 역할과 phase 의미 차이·지원 부족도 보고 |

1순위를 [실험 30 계획](EXPERIMENT30_PLAN.md)으로 구체화한다. 객체 선택·관측 mask와 기존 phase 중심을 유지하고 관계 평균의 track 경계 처리만 바꾼 후, 정상 모델을 다시 적합해 영향을 평가한다. 나머지 후보는 확정된 후속 실험이 아니다.

## 검증과 재현

105개 테스트 통과. 정상 25개 파일·2,417개 전이의 count/고유 영상 수/holdout과 mixed window를 독립 반복문으로 검증했다. 정상 source frame 경로·선택 bbox/track·phase/mask/descriptor/margin을 확인했고, 기존 특징·정상 모델·reference·점수 **181개 파일의 hash가 유지**됐다. 사례 선정은 이미지 검토 전에 동결됐고 30개 모두 검토 메모가 있다. 원본 test 입력/라벨과 기존 모델은 이번 실행에서 변경하지 않았다.

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=.cache/matplotlib
.venv/bin/python scripts/prepare_transition_provenance.py
.venv/bin/python scripts/audit_transition_provenance.py
.venv/bin/python scripts/render_transition_cases.py
.venv/bin/python scripts/diagnose_transition_history.py
# local contact sheets를 점검하고 exploratory notes를 기록; semantic GT로 사용하지 않음
.venv/bin/python scripts/validate_transition_provenance.py
.venv/bin/python scripts/plot_transition_provenance.py
.venv/bin/python -m pytest -q
```

사전 hash는 원본 실행 provenance다. 다른 환경에서는 입력 경로와 새로운 실행 기록이 필요하다. 계획 문서는 작성 당시 상태를 유지하고 완료 상태는 본 보고서/README를 따른다. contact sheet·원본·상세 특징 trace는 GitHub에 올리지 않는다.

[정상 coverage 집계](../results/experiment29/coverage_audit.json) · [사례 선정](../results/experiment29/case_selection.json) · [평균 이력 진단](../results/experiment29/history_diagnostic.json) · [사전 protocol](../results/experiment29/pre_audit_protocol.json) · [시각 검토 전 동결](../results/experiment29/pre_visual_checkpoint.json) · [local sheet 목록](../results/experiment29/local_contact_sheet_manifest.json) · [검증](../results/experiment29/validation.json) · [설정](../configs/experiment29.json)
