# Environment Design Best Practices

## General
* It is often helpful to being with the simplest version of the problem, to ensure the agent can learn it. From there increase
complexity over time.
* When possible, It is often helpful to ensure that you can complete the task by using a Player Brain to control the agent.

## Rewards
* The magnitude of any given reward should typically not be greater than 1.0 in order to ensure a more stable learning process.
* Positive rewards are often more helpful to shaping the desired behavior of an agent than negative rewards.
* For locomotion tasks, a small positive reward (+0.1) for forward progress is typically used. 
* If you want the agent the finish a task quickly, it is often helpful to provide a small penalty every step (-0.1). 

## States
* The magnitude of each state variable should be normalized to around 1.0. 
* States should include all variables relevant to allowing the agent to take the optimally informed decision.
* Categorical state variables such as type of object (Sword, Shield, Bow) should be encoded in one-hot fashion (ie `3` -> `0, 0, 1`).

## Actions
* When using continuous control, action values should be clipped to an appropriate range.
