# MPPI 실행

프로젝트 루트에서 실행합니다.

```powershell
python -m mppi.run
```

기본값은 1024개 후보, 75스텝 예측, 900스텝 실행입니다.
시뮬레이션 시간 30초이며 실제 계산 시간은 더 길 수 있습니다.
예측과 환경에 같은 25개 기저 평균 동역학을 사용하므로 독립 차량 검증은 아닙니다.

짧은 실행과 트랙/속도 변경:

```powershell
python -m mppi.run --steps 30 --samples 64
python -m mppi.run --radius 3 --half-width 0.75 --target-speed 1
```

## 비용과 구동기

수식과 가정은 [FORMULATION.md](FORMULATION.md)에 정리했습니다.
R, 기본 잡음, 탐색 잡음, lambda, nu를 구분하여 평균 이동 및 분산 변경 보정을 계산합니다.
임의의 importance_coefficient는 제거했습니다. 원래 Gaussian 잡음으로 보정/업데이트하고
입력 포화는 동역학에서 처리합니다.

구동기 응답 `da/dt = 10 * (u_des - a)`는 예측과 실행 양쪽에서 같은 함수로 계산합니다.
상태 순서는 차량 상태 7개 뒤에 실제 조향과 throttle을 붙인 9개입니다.
controller에는 이 9개 상태를 전달합니다. Simulator의 7차원 초기 상태는
config의 initial_applied 값으로 확장하며, 실제 입력을 안다면 9차원 초기 상태를 지정합니다.
학습 모델의 입력을 실제 적용 입력으로 취급하는 가정은 독립 데이터에서 검증해야 합니다.

가중치 정규화 온도와 보정식의 lambda는 별개입니다. 모드는
nominal_cost, fixed, theoretical입니다. 기본값은 nominal_cost와 alpha=20이며,
논문의 alpha=10과 차량 조건에서 조정한 결과를 paper_alignment 폴더에 저장합니다.

```powershell
python -m mppi.run --weight-temperature-mode nominal_cost --cost-normalization-alpha 10
python -m mppi.run --weight-temperature-mode fixed --temperature 0.3
```

`--disable-actuator`는 구동기 효과를 분리하는 실험용 옵션입니다.
현재 모델 사용 범위 검사 때문에 정지 상태에서는 시작하지 않습니다.
추가 옵션은 `python -m mppi.run --help`로 확인합니다.

## 저장 결과

실행마다 `mppi/results/실행시각/`에 새 폴더를 만듭니다.

| 파일 | 내용 |
|---|---|
| config.json | 설정, 상태 이름, 유도된 R/C0/nu/Bc |
| states.csv | 시간, 9개 상태, 정규화 트랙 오차 |
| controls.csv | desired/applied 입력, 유효 후보 비율, fallback, 계산 시간 |
| trajectory.npz | 상태, desired/applied 입력, 선택된 예측 궤적 |
| summary.json | 종료 원인, 주행 길이, RMSE, 계산 시간 |
| overview.png | 궤적, 속도, 상태, desired/applied 입력 |
| diagnostics.csv | 반복별 비용 통계, ESS, 최대 가중치, 탈락 원인 |
| candidate_diagnostics.npz | 반복별 후보 비용과 가중치, 탈락 마스크 |
| diagnostics.png | 비용, 가중치 집중도, 탈락 수 |

S개 입력에 대해 상태는 `(S+1,9)`입니다. controls[t]는 구동기에 전달한 desired 입력이고,
applied_controls[t]는 states[t]에서 states[t+1]로 진행할 때 차량 적분에 사용한 입력입니다.
동시 Euler 적분에서는 applied_controls[t]가 states[t,7:9]와 같습니다.
predictions[t,0]은 states[t]이며 예측 상태에도 구동기 상태가 포함됩니다.

비용 관계는 다음과 같습니다. 모든 값은 실제 적용 스케일입니다.

    trajectory_cost = state_cost + control_cost
    importance_cost = mean_shift_cost + variance_change_cost
    total_cost = trajectory_cost + importance_cost

control_cost는 현재 nominal의 제어 에너지로 한 번만 포함합니다.
effective_temperature는 해당 반복의 실제 가중치 정규화 온도입니다.
비용 통계는 최종 비용이 유한한 유효 후보 기준입니다.
mean/std/min/max는 분포 통계, best는 최저 총비용 후보의 값,
weighted_mean은 실제 가중치를 적용한 평균입니다.

ESS가 1에 가깝고 max_weight가 1에 가까우면 한 후보에 가중치가 집중된 상태입니다.
ESS만으로 주행 성능을 판단하지 않습니다. 탈락 원인은 중복 집계합니다.
상태 범위 검사에서 제외한 후보의 항별 비용은 NaN일 수 있습니다.
reject_total_nonfinite_count는 발산뿐 아니라 모든 비유한 최종 비용을 포함합니다.

step은 0부터 시작하는 입력 적용 번호, iteration은 해당 제어 안의 최적화 반복입니다.
실패한 마지막 제어 시도도 input_applied=False로 저장하므로 진단 행 수와 입력 수는 다를 수 있습니다.
유효 후보가 없으면 가중치는 0, ESS와 최대 가중치는 빈칸/JSON null로 저장합니다.

Ctrl+C와 유효 후보 없음은 완료된 기록을 저장하고 종료합니다.
시간 제한 도달은 정상 실행 종료이며 정해진 바퀴 수 완주를 뜻하지 않습니다.
검증은 `python -m unittest mppi.test_mppi -v`로 실행합니다.
변경 전 코드는 `results/paper_alignment/baseline_source.json`에 보존했습니다.
