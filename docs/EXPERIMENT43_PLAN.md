# 실험43 계획 — 공정별 learned phase head

상태: **계획, 학습·결과는 아직 없음**. 실험42의 추천순1만 구체화한다.

42의 공유 metric adapter는 weak validation loss를 낮췄지만 추가 연결은 정상 1/test 2건, phase 변경은 test 1 sample뿐이며 경보 개선은 없었다. R04 정상 phase3 편중 89.9%와 dwell unavailable도 그대로다. 다음은 사용자 지정 학습 순서에 따라 phase 관측 자체를 분리 검증한다. ReID 결과의 원인을 phase로 확정한다는 뜻은 아니다.

Detector41과 기존 IoU tracker를 고정하고, 공유된 동결 auxiliary visual feature 위에 R01/R02/R03/R04별 phase head를 학습한다. Process별 linear probe와 저랭크 residual adapter + classifier를 대조하여 head 추가와 표현 적응을 구분한다. Shared backbone·detector·ReID를 동시에 재학습하지 않고 appearance 8개 arm을 유지한다. Transition/dwell/normal appearance/calibration은 공정별 통계 모델로 재적합하며 최소 dwell 지원10을 유지한다.

먼저 정상 train70/val19의 phase target 지원·의미·모호성을 감사한다. 기존 latent phase를 실제 action GT로 부르거나 R04의 편중된 pseudo-label을 그대로 복제하지 않는다. 검수 가능한 정상 anchor·기하 근거·짧은 sequence로 weak target을 정의하고 불확실한 구간을 제외한다. Target을 새로 정의하면 기존 phase와의 비교 한계를 기록한다. 사람 action annotation이 없으므로 weak-label agreement·점유·경계 진단으로 한정한다. 지원을 확인한 뒤 입력·head·rank·loss·epoch·seed·선택 기준을 학습 전에 고정한다.

Checkpoint는 normal val로만 선택하고 calibration22는 optimizer와 checkpoint 선택에서 제외한다. Test 영상의 총길이·상대 진행률·미래 frame을 head 입력으로 쓰지 않는다. 기존 phase control 재현 → linear probe → low-rank head의 정상 관측 및 완결 run 지원 검증 → 모델·threshold 고정 → test 평가 순서로 진행한다. Phase 관측률을 정확도로 대체하지 않고 visual-only/combined·FPR/recall·events·dwell 가용성을 함께 보고한다. 지원 없는 phase를 억지로 학습하거나 test 결과에 맞춰 grammar를 바꾸지 않는다.

43의 결과·의의·보완점·추천순3개를 README와 보고서에 정리해 GitHub에 업로드한 뒤, 필요한 후속 변경을 결정한다. 그 이후 실험이나 논문 주제는 미리 정하지 않는다.

## 정상 표적 감사 후 확정한 학습안

첫 표적 감사는 보존했다. R02 텍스트 teacher에서 cosine gap≥0.01인 train 표적은 한 상태17개, val은 같은 상태8개뿐이었다. 검수한 강한 후보4개 중3개는 instrument-present 설명과 실제 빈 플랫폼이 불일치했다. 기준을 낮추어 그대로 학습하지 않고, R02 정상 train19영상의 platform–scissor 기하 관계를 사용한 별도 latent teacher로 교체했다. 따라서 R02의 새 상태는 기존 semantic4단계의 정답이 아니다.

R01은 기존 관측/이동 축/허용 band를 유지하고 정상 train의 위치 중심3개만 재적합한다. R03/R04는 기존 선택/gate/scaler를 유지하고 현재 detector 관측의 관계 중심4개만 train 영상으로 재적합한다. R04는 asinh 좌표를 유지한다. R02는 train에서 area gate·median/IQR·관계 중심을 적합한다. 이 **teacher 재적합 대조**를 기존41 control과 분리하여, teacher 변경과 head 학습을 혼동하지 않는다. 모든 target은 상대 거리 차≥0.1인 관측만 사용한다. Train 상대 위치는 R02~R04 cluster ID를 순열 정렬할 때만 쓰고 추론 입력으로 사용하지 않는다.

새 anchor60프레임을 Codex가 시각 검수했다. R01은 공간 위치, R02~R04는 detector의 기하 관계 군집이라는 범위만 인정한다. 군집 안의 instrument 유무, 운반 동작, blade 상태가 섞여 있어 action GT로 부르지 않는다. R02 state0의 val은6samples/2videos, R04 state0은16samples/2videos로 희소하다. 목표는 이런 weak target에서 process별 head를 학습하고 그 한계를 드러내는 것이다.

입력은 공통 frozen CLIP의 global + role0/1/2 최고 confidence crop을 연결한2048차원이며, 없는 역할은0이다. 현재 및 이전2sample의 trailing mean을 정규화한다. Teacher phase·좌표·frame index·총길이·미래 관측은 입력하지 않는다. Process마다 linear classifier, rank8 residual adapter+classifier를 별도로 학습한다. 두 branch의 초기 classifier와 sample 순서는 같다. Seed42/FP32,30epochs×32steps,batch128,class-balanced sampling,AdamW1e-3/wd1e-4이다. CE에 adapter의 원래 표현 보존항0.1을 더하며, 두 branch 모두 normal val의 class-macro CE로 epoch0부터 checkpoint를 선택한다. Calibration22와 test는 선택에서 제외한다.

추론은 teacher 관측이 유효한 현재 sample에서 head argmax를 사용하고, 관측이 없으면 이전 상태를 유지한다(초기0). Threshold나 새로운 smoothing을 test 결과로 고르지 않는다. Appearance8arms, 공정별 통계 모델, 최소 지원10과 R04 unavailable 정책을 유지한다. 독립 action GT가 없으므로 weak-label agreement·phase 점유·전이·dwell 지원과 최종 AD 지표를 구분해서 보고한다.
