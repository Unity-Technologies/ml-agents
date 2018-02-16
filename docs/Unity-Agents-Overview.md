# ML Agents Overview

<!-- 
Temp outline 
* What is ML Agents
* How is it related to Unity
* What are the elements
* How do they work together
* What are the features
* What are the use cases
-->

The ML Agents project makes machine-learning-based artifical intelligence available in Unity. Using the project you can populate simulations and games with _intelligent_ agents. ML Agents focuses on [reinforcement learning](link), but you can include traditional forms of AI and even extend the machine learning implementation with alternate learning techniques.

As illustrated below, ML Agents uses a few classes to orchestrate the behavior of one or more agents in a scene.   

![Typical ML Agents Scene Block Diagram](images/agents_diagram.png)

_An example of how a scene containing agents might be configured._

The main objects within any ML Agents environment are:

* Agents - Each agent instance makes its own observations and takes unique actions within the environment. An agent passes its observations to the brain it is linked to and receives the action to take in return. While training, an agent is also responsible for assigning a reward that estimates the value of the current state.
* Brains - A brain encapsulates the logic for making decisions for agents. An agent passes its observations of the environment to the brain and gets an action in return. Brains can be set to one of four modes:
    * External - Decisions are made using TensorFlow (or your machine learning library of choice) through communication over an open socket with our Python API. 
    * Internal - Decisions are made using a trained model embedded into the project via TensorFlowSharp. 
    * Player - Decisions are made using player input.
    * Heuristic - Decisions are made using hand-coded behavior.
* Academy - The Academy object within a scene orchestrates the observation and decision making process. It also contains all Brain objects within the environment as children.
* ExternalCommunicator - Handles communication between the Academy and external training, observation, or data collection processes, such as the ML Agents TensorFlow reinforcement learning module.

The states and observations of all agents with brains set to **External** are collected by the External Communicator, and communicated to an external process. By setting multiple agents to a single brain, actions can be decided in a batch fashion, taking advantage of the inherently parallel computations of neural networks. For more information on how these objects work together within a scene, see [Reinforcement Learning in Unity](Reinforcement-Learning-in-Unity.md).

For a walk through of the setup and training process, see [Getting Started](Getting-Started-with-Balance-Ball.md) and [Making a new Unity Environment](Making-a-New-Unity-Environment). For more in-depth look at how to create trainable agents, see [Reinforcement Learning in Unity](Reinforcement-Learning-in-Unity.md) and to learn about the training process itself, see [Training](Training-ML-Agents.md).  

