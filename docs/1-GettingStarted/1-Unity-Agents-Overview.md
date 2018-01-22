# Learning Environments Overview

![diagram](../images/agents_diagram.png)

A visual depiction of how a Learning Environment might be configured within ML-Agents.

The three main kinds of objects within any Agents Learning Environment are:

* Agent - Each Agent can have a unique set of states and observations, take unique actions within the environment, and can receive unique rewards for events within the environment. An agent's actions are decided by the brain it is linked to.
* Brain - Each Brain defines a specific state and action space, and is responsible for deciding which actions each of its linked agents will take. Brains can be set to one of four modes:
    * External - Action decisions are made using TensorFlow (or your ML library of choice) through communication over an open socket with our Python API. 
    * Internal (Experimental) - Actions decisions are made using a trained model embedded into the project via TensorFlowSharp. 
    * Player - Action decisions are made using player input.
    * Heuristic - Action decisions are made using hand-coded behavior.
* Academy - The Academy object within a scene also contains as children all Brains within the environment. Each environment contains a single Academy which defines the scope of the environment, in terms of:
    * Engine Configuration - The speed and rendering quality of the game engine in both training and inference modes.
    * Frameskip - How many engine steps to skip between each agent making a new decision.
    * Global episode length - How long the the episode will last. When reached, all agents are set to done.

The states and observations of all agents with brains set to External are collected by the External Communicator, and communicated via the Python API. By setting multiple agents to a single brain, actions can be decided in a batch fashion, taking advantage of the inherently parallel computations of neural networks. For more information on how these objects work together within a scene, see our wiki page.

## Flexible Training Scenarios

With the Unity ML-Agents, a variety of different kinds of training scenarios are possible, depending on how agents, brains, and rewards are connected. We are excited to see what kinds of novel and fun environments the community creates. For those new to training intelligent agents, below are a few examples that can serve as inspiration. Each is a prototypical environment configurations with a description of how it can be created using the ML-Agents SDK.

* **Single-Agent** - A single agent linked to a single brain. The traditional way of training an agent. An example is any single-player game, such as Chicken. [Video Link](https://www.youtube.com/watch?v=fiQsmdwEGT8&feature=youtu.be).
* **Simultaneous Single-Agent** - Multiple independent agents with independent reward functions linked to a single brain. A parallelized version of the traditional training scenario, which can speed-up and stabilize the training process. An example might be training a dozen robot-arms to each open a door simultaneously. [Video Link](https://www.youtube.com/watch?v=fq0JBaiCYNA).
* **Adversarial Self-Play** - Two interacting agents with inverse reward functions linked to a single brain. In two-player games, adversarial self-play can allow an agent to become increasingly more skilled, while always having the perfectly matched opponent: itself. This was the strategy employed when training AlphaGo, and more recently used by OpenAI to train a human-beating 1v1 Dota 2 agent.
* **Cooperative Multi-Agent** - Multiple interacting agents with a shared reward function linked to either a single or multiple different brains. In this scenario, all agents must work together to accomplish a task than couldn’t be done alone. Examples include environments where each agent only has access to partial information, which needs to be shared in order to accomplish the task or  collaboratively solve a puzzle. (Demo project coming soon)
* **Competitive Multi-Agent** - Multiple interacting agents with inverse reward function linked to either a single or multiple different brains. In this scenario, agents must compete with one another to either win a competition, or obtain some limited set of resources. All team sports would fall into this scenario. (Demo project coming soon)
* **Ecosystem** - Multiple interacting agents with independent reward function linked to either a single or multiple different brains. This scenario can be thought of as creating a small world in which animals with different goals all interact, such a savanna in which there might be zebras, elephants, and giraffes, or an autonomous driving simulation within an urban environment. (Demo project coming soon)

## Additional Features

Beyond the flexible training scenarios made possible by the Academy/Brain/Agent system, ML-Agents also includes other features which improve the flexibility and interpretability of the training process.

* **Monitoring Agent’s Decision Making** - Since communication in ML-Agents is a two-way street, we provide an Agent Monitor class in Unity which can display aspects of the trained agent, such as policy and value output within the Unity environment itself. By providing these outputs in real-time, researchers and developers can more easily debug an agent’s behavior.

* **Curriculum Learning** - It is often difficult for agents to learn a complex task at the beginning of the training process. Curriculum learning is the process of gradually increasing the difficulty of a task to allow more efficient learning. ML-Agents supports setting custom environment parameters every time the environment is reset. This allows elements of the environment related to difficulty or complexity to be dynamically adjusted based on training progress. 

* **Complex Visual Observations** - Unlike other platforms, where the agent’s observation might be limited to a single vector or image, ML-Agents allows multiple cameras to be used for observations per agent. This enables agents to learn to integrate information from multiple visual streams, as would be the case when training a self-driving car which required multiple cameras with different viewpoints, a navigational agent which might need to integrate aerial and first-person visuals, or an agent which takes both a raw visual input, as well as a depth-map or object-segmented image.
		
* **Imitation Learning (Coming Soon)** - It is often more intuitive to simply demonstrate the behavior we want an agent to perform, rather than attempting to have it learn via trial-and-error methods. In a future release, ML-Agents will provide the ability to record all state/action/reward information for use in supervised learning scenarios, such as imitation learning. By utilizing imitation learning, a player can provide demonstrations of how an agent should behave in an environment, and then utilize those demonstrations to train an agent in either a standalone fashion, or as a first-step in a reinforcement learning process.
