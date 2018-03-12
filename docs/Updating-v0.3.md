# Migrating from v0.2 to v0.3

There area number of new features and improvements in ML-Agents v0.3 which change both the training process and Unity API in ways which will cause imcompatabilities with environments made using older versions. This page is designed to highlight those changes for users familiar with v0.2 in order to make the trainsition simple.

## Important
 * ML-Agents no longer supports usage with Python 2. 

## Python Training
 * The `learn.py` script now replaces `ppo.py` and the `PPO.ipynb` python notebook as the launching point for training with ML-Agents. For more information on using `learn.py`, see [here]().
 * Hyperparameters for training brains are now stored in the `trainer_config.yaml` file. For more information on using this file, see [here]().

## Unity API
 * Modifications to an Agent's rewards must now be done using either `AddReward()` or `SetReward()`.
 * Setting an Agent to done now requires the use of the `Done()` method.
 * `CollectStates()` has been replaced by `CollectObservations()`, which now no longer returns a list of floats.
 * To collect observations, call `AddVectorObs()` within `CollectObservations()`. Note that you can call `AddVectorObs()` with floats, integers, lists and arrays of floats, Vector3 and Quaternions. 
 * `AgentStep()` has been replaced by `AgentAction()`.
 * `WaitTime()` has been removed.
 * The `Frame Skip` field of the Academy is replaced by the Agent's `Decision Frequency` field, enabling agent to make decisions at different frequencies.

## Semantics
* _State_ is now referred to as _Vector Observation_.
* _Observation_ is now referred to as _Visual Observation_.
* _Action_ is now referred to as _Vector Action_.