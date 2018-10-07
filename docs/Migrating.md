# Migrating

## Migrating from ML-Agents toolkit v0.4 to v0.5

### Important

* The Unity project `unity-environment` has been renamed `UnitySDK`.
* The `python` folder has been renamed to `ml-agents`. It now contains two
  packages, `mlagents.env` and `mlagents.trainers`. `mlagents.env` can be used
  to interact directly with a Unity environment, while `mlagents.trainers`
  contains the classes for training agents.
* The supported Unity version has changed from `2017.1 or later` to `2017.4
  or later`. 2017.4 is an LTS (Long Term Support) version that helps us
  maintain good quality and support. Earlier versions of Unity might still work,
  but you may encounter an
  [error](FAQ.md#instance-of-corebraininternal-couldnt-be-created) listed here.

### Unity API

* Discrete Actions now use [branches](https://arxiv.org/abs/1711.08946). You can
  now specify concurrent discrete actions. You will need to update the Brain
  Parameters in the Brain Inspector in all your environments that use discrete
  actions. Refer to the
  [discrete action documentation](Learning-Environment-Design-Agents.md#discrete-action-space)
  for more information.

### Python API

* In order to run a training session, you can now use the command
  `mlagents-learn` instead of `python3 learn.py` after installing the `mlagents`
  packages. This change is documented
  [here](Training-ML-Agents.md#training-with-mlagents-learn). For example,
  if we previously ran

  ```sh
  python3 learn.py 3DBall --train
  ```

  from the `python/` directory, we now run

  ```sh
  mlagents-learn config/trainer_config.yaml --env=3DBall --train
  ```

  from the directory where we installed the ML-Agents Toolkit.

* It is now required to specify the path to the yaml trainer configuration file
  when running `mlagents-learn`. For an example trainer configuration file, see
  [trainer_config.yaml](../config/trainer_config.yaml). An example of passing
  a trainer configuration to `mlagents-learn` is shown above.
* The environment name is now passed through the `--env` option.
* Curriculum learning has been changed. Refer to the
    [curriculum learning documentation](Training-Curriculum-Learning.md)
    for detailed information. In summary:
  * Curriculum files for the same environment must now be placed into a folder.
    Each curriculum file should be named after the brain whose curriculum it
    specifies.
  * `min_lesson_length` now specifies the minimum number of episodes in a lesson
    and affects reward thresholding.
  * It is no longer necessary to specify the `Max Steps` of the Academy to use
    curriculum learning.

## Migrating from ML-Agents toolkit v0.3 to v0.4

### Unity API

* `using MLAgents;` needs to be added in all of the C# scripts that use
  ML-Agents.

### Python API

* We've changed some of the Python packages dependencies in requirement.txt
  file. Make sure to run `pip3 install .` within your `ml-agents/python` folder
  to update your Python packages.

## Migrating from ML-Agents toolkit v0.2 to v0.3

There are a large number of new features and improvements in the ML-Agents
toolkit v0.3 which change both the training process and Unity API in ways which
will cause incompatibilities with environments made using older versions. This
page is designed to highlight those changes for users familiar with v0.1 or v0.2
in order to ensure a smooth transition.

### Important

* The ML-Agents toolkit is no longer compatible with Python 2.

### Python Training

* The training script `ppo.py` and `PPO.ipynb` Python notebook have been
  replaced with a single `learn.py` script as the launching point for training
  with ML-Agents. For more information on using `learn.py`, see
  [here](Training-ML-Agents.md#training-with-mlagents-learn).
* Hyperparameters for training Brains are now stored in the
  `trainer_config.yaml` file. For more information on using this file, see
  [here](Training-ML-Agents.md#training-config-file).

### Unity API

* Modifications to an Agent's rewards must now be done using either
  `AddReward()` or `SetReward()`.
* Setting an Agent to done now requires the use of the `Done()` method.
* `CollectStates()` has been replaced by `CollectObservations()`, which now no
  longer returns a list of floats.
* To collect observations, call `AddVectorObs()` within `CollectObservations()`.
  Note that you can call `AddVectorObs()` with floats, integers, lists and
  arrays of floats, Vector3 and Quaternions.
* `AgentStep()` has been replaced by `AgentAction()`.
* `WaitTime()` has been removed.
* The `Frame Skip` field of the Academy is replaced by the Agent's `Decision
  Frequency` field, enabling the Agent to make decisions at different frequencies.
* The names of the inputs in the Internal Brain have been changed. You must
  replace `state` with `vector_observation` and `observation` with
  `visual_observation`. In addition, you must remove the `epsilon` placeholder.

### Semantics

In order to more closely align with the terminology used in the Reinforcement
Learning field, and to be more descriptive, we have changed the names of some of
the concepts used in ML-Agents. The changes are highlighted in the table below.

| Old - v0.2 and earlier | New - v0.3 and later |
| --- | --- |
| State | Vector Observation |
| Observation | Visual Observation |
| Action | Vector Action |
| N/A | Text Observation |
| N/A | Text Action |
