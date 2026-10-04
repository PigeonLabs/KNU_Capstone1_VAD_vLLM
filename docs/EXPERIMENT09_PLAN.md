# 다음 실험 계획 — 09

상태: 실험 08 결과에서 선택한 다음 단계. 아직 구현·실행하지 않았다.

## 질문과 근거

이전 상태별로 정상 전이 점수를 보정하면 전역 보정의 상태별 offset과 최종 결합의 약점을 줄일 수 있는가?

실험 08에서는 Visual/Process AUROC가 각각 0.7330/0.5520으로 올랐지만 Combined AUROC는 0.6361로 낮아졌다. 정상 calibration 전이는 대부분 self-transition인데 전역 percentile 중앙값은 이전 상태별 0.1899/0.5213/0.8115/0.9644였다. 이 진단은 조건부 보정을 검증할 근거이며, 성능 개선이나 실패 원인의 확정은 아니다.

## 다음 실행 범위

1. 실험 08의 관계 상태·모든 특징·분할·PCA·raw 전이 점수·Visual 점수·0.5/0.5 결합을 보존한다. 전이 순서 prior나 진행량 모듈을 동시에 바꾸지 않는다.
2. 정상 calibration의 t>0 전이를 이전 latent state별로 나누어 기존과 같은 midrank empirical CDF를 계산한다. 정상 support가 최소 10개인 상태에 조건부 CDF를 사용한다. 현재 support는 264/211/208/16개다. 지원 부족 상태와 이전 상태가 없는 첫 sample은 기존 전역 CDF로 fallback한다. 미관측 hold 정책은 실험 08 그대로 유지한다.
3. 최종 Combined q99만 새 점수의 정상 calibration에서 다시 정한다. 테스트 라벨로 가중치·support 기준·threshold를 선택하거나 여러 설정을 탐색하지 않는다.
4. 정상 상태별 process percentile 진단, Visual/raw 전이 동일성, fallback 빈도를 검사한다. 조건부 보정에서도 이산 전이와 동점 때문에 상태별 분포가 완전히 같아질 필요는 없다. calibration 자체에서 얻은 보정 특성은 일반화 성능과 구분한다.
5. 실험 08과 09의 Process/Combined AUROC/AP, 각 정상 q99의 정상 오탐/이상 recall, GT 구간별 미탐·지연을 비교한다. 동일 테스트 오탐률 비교가 아니라는 점을 유지한다. 작은 상태별 calibration support와 정상 프레임 상관에 따른 불안정성도 기록한다.

R03은 개발 장면이다. 결과를 본 뒤 다음 추천순 3개를 갱신하며 후속 실험 전체나 최종 논문 주제를 미리 확정하지 않는다.
