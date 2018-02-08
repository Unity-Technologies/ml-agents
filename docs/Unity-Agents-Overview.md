# ML Agents

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

![diagram](../images/agents_diagram.png)

_An example of how a scene containing agents might be configured._

The main objects within any ML Agents environment are:

* Agents - Each Agent instance makes its own observations and takes unique actions within the environment. An agent passes its observations to the brain it is linked to and receives the action to take in return. While training, an agent is also responsible for assigning a reward that estimates the value of the current state.
* Brains - A Brain encapsulates the logic for making decisions for agents. An agent passes its observations of the environment to the brain and gets an action in return. Brains can be set to one of four modes:
    * External - Decisions are made using TensorFlow (or your machine learning library of choice) through communication over an open socket with our Python API. 
    * Internal - Decisions are made using a trained model embedded into the project via TensorFlowSharp. 
    * Player - Decisions are made using player input.
    * Heuristic - Decisions are made using hand-coded behavior.
* Academy - The Academy object within a scene orchestrates the observation and decision making process. It also contains all Brains within the environment as children.
* ExternalCommunicator - Handles communication between the Academy and external training, observation, or data collection processes, such as the ML Agents TensorFlow reinforcement learning module.

The states and observations of all agents with brains set to External are collected by the External Communicator, and communicated to an external process. ML Agents includes  the Python API. By setting multiple agents to a single brain, actions can be decided in a batch fashion, taking advantage of the inherently parallel computations of neural networks. For more information on how these objects work together within a scene, see [Reinforcement Learning in Unity](link).

## Setting up ML Agents

To use ML Agents, follow the [installation instructions](link) and then do the following in your Unity project:

1. Create an environment for your agents to live in. An environment can range from a simple physical simulation containing a few objects to an entire game or ecosystem.       
2. Implement an Academy subclass and add it to a GameObject in the Unity scene containing the environment. This GameObject will serve as the parent for any Brain objects in the scene. Your Academy class can implement a few optional methods to update the scene independently of any agents. For example, you can add, move, or delete agents and other entities in the environment.
3. Add one or more Brain objects to the scene as children of the Academy.
4. Implement your Agent subclasses. An Agent subclass defines the code an agent uses to observe its environment, carryout assigned actions, and calculates the reward used for reinforcement training. You can also implement optional methods to reset the agent when it has finished or failed its task.
5. Add your Agent subclasses to appropriate GameObjects, typically, the object in the scene that represents the agent in the simulation. Each Agent object must be assigned a Brain.
6. If training, set the Brain type to External and [run the training process](link).  

For a walk through of the prcoess, see [Getting Started](link) and [Making a new Unity Environment](link). For more in-depth look at how to create trainable agents, see [Reinforcement Learning in Unity](link) and to learn about the training itself, see [Training](link).  

<!-- 
Editor's Note: the rest of this document still needs revising.
 -->

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
