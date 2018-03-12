# Updating from v0.2 to v0.3 of ML-Agents

## Important
 * ML-Agents will no longer support Python 2. 

## Training
 * The script `learn.py` now replaces `ppo.py` and the `PPO.ipynb` python notebook.
 * Training hyperparameters are now given to trainers via the `trainer_config.yaml` file.

## Environment
 * Modifications to an Agent's rewards must be done via `AddReward()` or `SetReward()`.
 * Setting an Agent to done now requires the use of the `Done()` method.
 * `CollectStates()` is replaced by `CollectObservations()`.
 * To collect observations, call `AddVectorObs()` within `CollectObservations()`. Note that you can call `AddVectorObs()` with floats, integers, lists and arrays of floats, Vector3 and Quaternions. 
 * `AgentStep()` is replaced by `AgentAct()`.
 * `WaitTime()` is deprecated.
 * The `Frame Skip` field of the Academy is replaced by the Agent's `Decision Frequency` field.
