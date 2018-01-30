# Reinforcement Learning in Unity

Reinforcement learning is an artificial intelligence technique that trains agents to perform tasks by rewarding desireable behavior. During reinforcement learning, an agent explores its environment, observes the state of things, and, based on that state takes an action. If the action leads to a better state, the agent receives a positive reward. If it leads to a less desirable state, then the agent receives no reward -- or sometimes a negative reward (punishment). As the agent learns during training, it optimizes its decision making so that it receives the maximum reward over time.

ML Agents uses a reinforcement learning technique called [Proximal Policy Optimization (PPO)](link). PPO uses a neural network to approximate the ideal function that maps an agent's observations to the best action an agent can take in a given state. The ML Agents PPO algorithm is implemented in TensorFlow and runs in a separate Python process (communicating with the running Unity environment over a socket). 

**Note:** if you aren't studying machine and reinforcement learning as a subject and just want to train agents to accomplish tasks, you can treat PPO training as a _black box_. There are a few training-related parameters to adjust inside Unity as well as on the Python training side, but you do not need in-depth knowledge of the algorithm itself to successfully create and train agents. (Step-by-step procedures for running the training process are provided in the [Training  section](link). 

This document discusses how to design your agents to successfully learn. It also covers the things to consider when creating a new environment for training agents or modifying an existing scene to serve as a training environment. 

## Agents

An agent is an actor in some sort of environment that can observe its world and decide on the best course of action using those observations. You create agents by extending the [Agent](link) class. 

<!-- We could discuss more about the arbitrary line between actor and environment here, but it can wait until we have examples of multi-agent entities (like a executive brain setting goals for a "robot" and sub-brains moving the robot limbs to accomplish these goals). -->

In the ML Agents framework, an agent passes its observations to its brain at each simulation step. The brain, then, makes a decision and passes the chosen action back to the agent. The agent both carries out the action, for example, moves in one direction or another, and also calculates a reward based on the current state. In training, the reward is used to discover the optimal decision-making policy, so the reward assignment is a critical component in successful training. (The reward is not used by already trained agents.)

How a brain makes its decisions depends on the type of brain it is. In training, an External brain adjusts its internal policy parameters to optimize the rewards received over time. An Internal brain uses the trained policy parameters to make decisions (and no longer adjusts the parameters in search of a better decision). 

The other types of brains do not directly involve training, but you might find them useful as part of a training project. A Heuristic brain provides a hand-programmed decision making process, for example you could program a decision tree. You can use agents with heuristic brains as part of an environment, to help test an environment, to compare a trained agent to a hard-coded agent. You can even use a hybrid system where you use a heuristic brain in some states and a trained brain in others. A Player brain maps actions to keyboard keys so that a human can control an agent directly. Player brains are useful when testing an environment before training.

  
### State Observations

To make decisions, an agent must observe its environment to determine its current state. A state observation can take the following forms:

* Continuous: a feature vector consisting of an array of numbers. 
* Camera observation: one or more camera images.
* Discrete: an index into a state table (typically only useful for the simplest of environments).

The main choice to make when creating an agent is whether to use a feature vector or a camera image as the agent's observation. Discrete state observations can only be used when the state space can be described as a table, which is typically only practical for very simple cases when you consider that even a simple game like [tic-tac-toe has at least 765 possible states](https://en.wikipedia.org/wiki/Game_complexity).

Agents using feature vectors are typically more efficient and faster to train than those using camera observations, but they also require more detailed knowledge of the internal workings of the simulation or game. Conversely, agents using camera images can capture state of arbitrary complexity and are useful when the state is difficult to describe numerically. However, agents using camera images require more computations per decision and are typically slower to train.  

When you create an Agent subclass, implement the [CollectState()](link) method to create the feature vector. The Brain object assigned to an agent uses the feature vector to decide which of the possible actions to take. 

#### Continuous State -- Feature Vectors

##### Normalization


#### Camera Observations

#### Discrete State -- Table Lookup


### Rewards

### Actions

### Agent Brains

### Implementing an Agent Subclass

## Environments



