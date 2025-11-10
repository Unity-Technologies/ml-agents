[한국어 버전 보기 (View Korean Version)](README.ko.md)

# UR16e Pick and Place with Unity ML-Agents

## Project Overview

This project aims to train a UR16e robotic arm to perform a 'Pick and Place' task using the Unity ML-Agents toolkit. The robot learns through reinforcement learning to move to a specified target object.

This project is implemented based on the [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) official repository.

## Project Structure

- **Assets/UR16 agents.unity**: The main Unity scene file. You can run agent training and simulation in this scene.
- **Assets/IK_toolkit/Scripts/UR16Agent.cs**: The core script defining the agent's behavior, observations, and reward logic.
- **Assets/IK_toolkit/Prefabs/RobotSet.prefab**: A prefab containing the UR16e robot and the training environment.
- **Assets/*.onnx**: Trained neural network model files. Used by Unity's Barracuda inference engine.

## Getting Started

### Prerequisites

- **Unity Editor**: Version `6000.0.42f1` (or a compatible version)
- **Unity ML-Agents**: The package version included in the project (refer to `Packages/manifest.json`)

### How to Run

1.  Open this project via Unity Hub.
2.  Open the `Assets/UR16 agents.unity` scene.
3.  Press the Play button in the Unity Editor to start the simulation.
4.  If using a trained model, you need to assign the `.onnx` file to the `Model` field of the `UR16Agent` component.

## Agent Details (`UR16Agent.cs`)

### Observations

The agent perceives its state using a total of 6-dimensional vector observations.

- Local position of `ikTarget` (Vector3, 3 dimensions)
- Local position of `targetObject` (Vector3, 3 dimensions)

### Actions

The agent outputs 3 continuous action values, each between -1 and 1. These values determine the movement direction of the `ikTarget` in 3D space.

- `action[0]`: X-axis movement
- `action[1]`: Y-axis movement
- `action[2]`: Z-axis movement

### Rewards

- **Target Reached**: If the distance between `ikTarget` and `targetObject` is less than 0.05 units, a reward of `+10.0` is given, and the episode ends.
- **Movement Limit Penalty**: If the `ikTarget` moves too far from its origin (exceeds 2.1 units) or too close (less than 0.4 units), a penalty of `-0.1` is given, and the episode ends.
- **Collision Penalty**: If the `ikTarget` collides with another object (detected by TouchSensor), a penalty of `-0.1` is given, and the episode ends.
- **Time Penalty**: A small penalty of `-0.003` is given at each step to encourage the agent to reach the target quickly.

### Heuristic

You can control the agent directly using the keyboard for testing.

- **W/S**: Z-axis movement
- **A/D**: X-axis movement
- **Q/E**: Y-axis movement
