# 실험43 계획 — 공정별 learned phase head

상태: **계획, 학습·결과는 아직 없음**. 실험42의 추천순1만 구체화한다.

42의 공유 metric adapter는 weak validation loss를 낮췄지만 추가 연결은 정상 1/test 2건, phase 변경은 test 1 sample뿐이며 경보 개선은 없었다. R04 정상 phase3 편중 89.9%와 dwell unavailable도 그대로다. 다음은 사용자 지정 학습 순서에 따라 phase 관측 자체를 분리 검증한다. ReID 결과의 원인을 phase로 확정한다는 뜻은 아니다.

Detector41과 기존 IoU tracker를 고정하고, 공유된 동결 auxiliary visual feature 위에 R01/R02/R03/R04별 phase head를 학습한다. Process별 linear probe와 저랭크 residual adapter + classifier를 대조하여 head 추가와 표현 적응을 구분한다. Shared backbone·detector·ReID를 동시에 재학습하지 않고 appearance 8개 arm을 유지한다. Transition/dwell/normal appearance/calibration은 공정별 통계 모델로 재적합하며 최소 dwell 지원10을 유지한다.

먼저 정상 train70/val19의 phase target 지원·의미·모호성을 감사한다. 기존 latent phase를 실제 action GT로 부르거나 R04의 편중된 pseudo-label을 그대로 복제하지 않는다. 검수 가능한 정상 anchor·기하 근거·짧은 sequence로 weak target을 정의하고 불확실한 구간을 제외한다. Target을 새로 정의하면 기존 phase와의 비교 한계를 기록한다. 사람 action annotation이 없으므로 weak-label agreement·점유·경계 진단으로 한정한다. 지원을 확인한 뒤 입력·head·rank·loss·epoch·seed·선택 기준을 학습 전에 고정한다.

Checkpoint는 normal val로만 선택하고 calibration22는 optimizer와 checkpoint 선택에서 제외한다. Test 영상의 총길이·상대 진행률·미래 frame을 head 입력으로 쓰지 않는다. 기존 phase control 재현 → linear probe → low-rank head의 정상 관측 및 완결 run 지원 검증 → 모델·threshold 고정 → test 평가 순서로 진행한다. Phase 관측률을 정확도로 대체하지 않고 visual-only/combined·FPR/recall·events·dwell 가용성을 함께 보고한다. 지원 없는 phase를 억지로 학습하거나 test 결과에 맞춰 grammar를 바꾸지 않는다.

43의 결과·의의·보완점·추천순3개를 README와 보고서에 정리해 GitHub에 업로드한 뒤, 필요한 후속 변경을 결정한다. 그 이후 실험이나 논문 주제는 미리 정하지 않는다.
