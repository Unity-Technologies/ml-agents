# Environment Design Best Practices

## General

* It is often helpful to start with the simplest version of the problem, to
  ensure the agent can learn it. From there, increase complexity over time. This
  can either be done manually, or via Curriculum Learning, where a set of
  lessons which progressively increase in difficulty are presented to the agent
  ([learn more here](Training-Curriculum-Learning.md)).
* When possible, it is often helpful to ensure that you can complete the task by
  using a heuristic to control the agent. To do so, check the `Use Heuristic`
  checkbox on the Agent and implement the `Heuristic()` method on the Agent.
* It is often helpful to make many copies of the agent, and give them the same
  `Behavior Name`. In this way the learning process can get more feedback
  information from all of these agents, which helps it train faster.

## Rewards

* The magnitude of any given reward should typically not be greater than 1.0 in
  order to ensure a more stable learning process.
* Positive rewards are often more helpful to shaping the desired behavior of an
  agent than negative rewards.
* For locomotion tasks, a small positive reward (+0.1) for forward velocity is
  typically used.
* If you want the agent to finish a task quickly, it is often helpful to provide
  a small penalty every step (-0.05) that the agent does not complete the task.
  In this case completion of the task should also coincide with the end of the
  episode.
* Overly-large negative rewards can cause undesirable behavior where an agent
  learns to avoid any behavior which might produce the negative reward, even if
  it is also behavior which can eventually lead to a positive reward.

## Vector Observations

* Vector Observations should include all variables relevant to allowing the
  agent to take the optimally informed decision.
* In cases where Vector Observations need to be remembered or compared over
  time, increase the `Stacked Vectors` value to allow the agent to keep track of
  multiple observations into the past.
* Categorical variables such as type of object (Sword, Shield, Bow) should be
  encoded in one-hot fashion (i.e. `3` > `0, 0, 1`).
* Besides encoding non-numeric values, all inputs should be normalized to be in
  the range 0 to +1 (or -1 to 1). For example, the `x` position information of
  an agent where the maximum possible value is `maxValue` should be recorded as
  `VectorSensor.AddObservation(transform.position.x / maxValue);` rather than
  `VectorSensor.AddObservation(transform.position.x);`. See the equation below for one approach
  of normalization.
* Positional information of relevant GameObjects should be encoded in relative
  coordinates wherever possible. This is often relative to the agent position.

![normalization](images/normalization.png)

## Vector Actions

* When using continuous control, action values should be clipped to an
  appropriate range. The provided PPO model automatically clips these values
  between -1 and 1, but third party training systems may not do so.
* Be sure to set the Vector Action's Space Size to the number of used Vector
  Actions, and not greater, as doing the latter can interfere with the
  efficiency of the training process.
