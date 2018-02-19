# Environment Design Best Practices

## General
* It is often helpful to start with the simplest version of the problem, to ensure the agent can learn it. From there increase
complexity over time. This can either be done manually, or via Curriculum Learning, where a set of lessons which progressively increase in difficulty are presented to the agent ([learn more here](../docs/curriculum.md)).
* When possible, it is often helpful to ensure that you can complete the task by using a Player Brain to control the agent.

## Rewards
* The magnitude of any given reward should typically not be greater than 1.0 in order to ensure a more stable learning process.
* Positive rewards are often more helpful to shaping the desired behavior of an agent than negative rewards.
* For locomotion tasks, a small positive reward (+0.1) for forward velocity is typically used. 
* If you want the agent to finish a task quickly, it is often helpful to provide a small penalty every step (-0.05) that the agent does not complete the task. In this case completion of the task should also coincide with the end of the episode.
* Overly-large negative rewards can cause undesirable behavior where an agent learns to avoid any behavior which might produce the negative reward, even if it is also behavior which can eventually lead to a positive reward.

## States
* States should include all variables relevant to allowing the agent to take the optimally informed decision.
* Categorical state variables such as type of object (Sword, Shield, Bow) should be encoded in one-hot fashion (ie `3` -> `0, 0, 1`).
* Besides encoding non-numeric values, all inputs should be normalized to be in the range 0 to +1 (or -1 to 1). For example rotation information on GameObjects should be recorded as `state.Add(transform.rotation.eulerAngles.y/180.0f-1.0f);` rather than `state.Add(transform.rotation.y);`. See the equation below for one approach of normaliztaion. 
* Positional information of relevant GameObjects should be encoded in relative coordinates wherever possible. This is often relative to the agent position.

![normalization](../images/normalization.png)

## Actions
* When using continuous control, action values should be clipped to an appropriate range.
* Be sure to set the action-space-size to the number of used actions, and not greater, as doing the latter can interfere with the efficency of the training process.
