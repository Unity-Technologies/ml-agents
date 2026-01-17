[View English Version](README.md)

# UR16e Pick and Place with Unity ML-Agents (Korean)

## 프로젝트 개요

이 프로젝트는 Unity ML-Agents 툴킷을 사용하여 UR16e 로봇 팔이 'Pick and Place' 작업을 수행하도록 학습시키는 것을 목표로 합니다. 로봇은 강화학습을 통해 지정된 목표 물체로 이동하는 방법을 학습합니다.

이 프로젝트는 [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) 공식 저장소를 기반으로 구현되었습니다.

## 프로젝트 구조

- **Assets/UR16 agents.unity**: 메인 Unity 씬 파일입니다. 이 씬에서 에이전트의 학습 및 시뮬레이션을 실행할 수 있습니다.
- **Assets/IK_toolkit/Scripts/UR16Agent.cs**: 에이전트의 행동, 관찰, 보상 로직이 정의된 핵심 스크립트입니다.
- **Assets/IK_toolkit/Prefabs/RobotSet.prefab**: UR16e 로봇과 학습 환경을 포함하는 프리팹입니다.
- **Assets/*.onnx**: 학습된 신경망 모델 파일입니다. Unity의 Barracuda 추론 엔진을 통해 사용됩니다.

## 시작하기

### 요구 사항

- **Unity Editor**: `6000.0.42f1` 버전 (또는 호환 가능한 버전)
- **Unity ML-Agents**: 프로젝트에 포함된 패키지 버전 (`Packages/manifest.json` 참조)

### 실행 방법

1.  Unity Hub를 통해 이 프로젝트를 엽니다.
2.  `Assets/UR16 agents.unity` 씬을 엽니다.
3.  Unity Editor에서 재생(Play) 버튼을 눌러 시뮬레이션을 시작합니다.
4.  학습된 모델을 사용하는 경우, `UR16Agent` 컴포넌트의 `Model` 필드에 `.onnx` 파일을 할당해야 합니다.

## 에이전트 상세 정보 (`UR16Agent.cs`)

### 관찰 (Observations)

에이전트는 총 6차원의 벡터 관찰 값을 사용하여 상태를 인식합니다.

- `ikTarget`의 로컬 위치 (Vector3, 3차원)
- `targetObject`의 로컬 위치 (Vector3, 3차원)

### 행동 (Actions)

에이전트는 3개의 연속적인 행동 값을 출력하며, 각 값은 -1과 1 사이입니다. 이 값들은 `ikTarget`의 3D 공간상 이동 방향을 결정합니다.

- `action[0]`: X축 이동
- `action[1]`: Y축 이동
- `action[2]`: Z축 이동

### 보상 (Rewards)

- **목표 도달**: `ikTarget`과 `targetObject` 사이의 거리가 0.05 유닛 미만이 되면 `+10.0`의 보상을 받고 에피소드가 종료됩니다.
- **이동 제한 페널티**: `ikTarget`이 로봇의 원점으로부터 너무 멀어지거나(2.1 유닛 초과) 가까워지면(0.4 유닛 미만) `-0.1`의 페널티를 받고 에피소드가 종료됩니다.
- **충돌 페널티**: `ikTarget`이 다른 물체와 충돌하면(TouchSensor 감지) `-0.1`의 페널티를 받고 에피소드가 종료됩니다.
- **시간 페널티**: 에이전트가 신속하게 목표에 도달하도록 유도하기 위해 매 스텝마다 `-0.003`의 작은 페널티를 받습니다.

### 휴리스틱 (Heuristic)

키보드를 사용하여 에이전트를 직접 조작해볼 수 있습니다.

- **W/S**: Z축 이동
- **A/D**: X축 이동
- **Q/E**: Y축 이동


### Vidoes
[![ML-Agent UR16 PPO 1](https://img.youtube.com/vi/hCeppvgj01s/0.jpg)](https://www.youtube.com/watch?v=hCeppvgj01s)
[![ML-Agent UR16 PPO 2](https://img.youtube.com/vi/eraQduLw8Zk/0.jpg)](https://www.youtube.com/watch?v=eraQduLw8Zk)
[![ML-Agent UR16 PPO 3](https://img.youtube.com/vi/wddv8uCz7a0/0.jpg)](https://www.youtube.com/watch?v=wddv8uCz7a0)
[![ML-Agent UR16 PPO 4](https://img.youtube.com/vi/cB5jqH2LSJE/0.jpg)](https://www.youtube.com/watch?v=cB5jqH2LSJE)
[![ML-Agent UR16 PPO 5](https://img.youtube.com/vi/NXXl5ug3gQI/0.jpg)](https://www.youtube.com/watch?v=NXXl5ug3gQI)
