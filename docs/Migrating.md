# Migrating from ML-Agents toolkit v0.3 to v0.4

## Unity API
 * `using MLAgents;` needs to be added in all of the C# scripts that use ML-Agents. 

## Python API
 * We've changed some of the python packages dependencies in requirement.txt file. Make sure to run `pip install .` within your `ml-agents/python` folder to update your python packages. 

# Migrating from ML-Agents toolkit v0.2 to v0.3

There are a large number of new features and improvements in the ML-Agents toolkit v0.3 which change both the training process and Unity API in ways which will cause incompatibilities with environments made using older versions. This page is designed to highlight those changes for users familiar with v0.1 or v0.2 in order to ensure a smooth transition.

## Important
 * The ML-Agents toolkit is no longer compatible with Python 2. 

## Python Training
 * The training script `ppo.py` and `PPO.ipynb` Python notebook have been replaced with a single `learn.py` script as the launching point for training with ML-Agents. For more information on using `learn.py`, see [here]().
 * Hyperparameters for training brains are now stored in the `trainer_config.yaml` file. For more information on using this file, see [here]().

## Unity API
 * Modifications to an Agent's rewards must now be done using either `AddReward()` or `SetReward()`.
 * Setting an Agent to done now requires the use of the `Done()` method.
 * `CollectStates()` has been replaced by `CollectObservations()`, which now no longer returns a list of floats.
 * To collect observations, call `AddVectorObs()` within `CollectObservations()`. Note that you can call `AddVectorObs()` with floats, integers, lists and arrays of floats, Vector3 and Quaternions. 
 * `AgentStep()` has been replaced by `AgentAction()`.
 * `WaitTime()` has been removed.
 * The `Frame Skip` field of the Academy is replaced by the Agent's `Decision Frequency` field, enabling agent to make decisions at different frequencies.
 * The names of the inputs in the Internal Brain have been changed. You must replace `state` with `vector_observation` and `observation` with `visual_observation`. In addition, you must remove the `epsilon` placeholder.

## Semantics
In order to more closely align with the terminology used in the Reinforcement Learning field, and to be more descriptive, we have changed the names of some of the concepts used in ML-Agents. The changes are highlighted in the table below.

| Old - v0.2 and earlier | New - v0.3 and later |
| --- | --- |
| State | Vector Observation |
| Observation | Visual Observation |
| Action | Vector Action |
| N/A | Text Observation |
| N/A | Text Action |
