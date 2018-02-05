# Agents

An agent is an actor that can observe its environment and decide on the best course of action using those observations. Create agents in Unity by extending the [Agent](link) class. The most important aspects of creating agents that can successfully learn are the observations the agent collects and the reward you assign to estimate the value of the agent's current state toward accomplishing its tasks.

In the ML Agents framework, an agent passes its observations to its brain at each simulation step. The brain, then, makes a decision and passes the chosen action back to the agent. The agent code executes the action, for example, it moves the agent in one direction or another, and also calculates a reward based on the current state. In training, the reward is used to discover the optimal decision-making policy. (The reward is not used by already trained agents.)

The Brain class abstracts out the decision making logic from the agent itself so that you can use the same brain in multiple agents. 
How a brain makes its decisions depends on the type of brain it is. An **External** brain simply passes the observations from its agents to an external process and then passes the decisions made externally back to the agents. During training, the ML Agents [reinforcement learning](1-Reinforcement-Learning-in_Unity) algorithm adjusts its internal policy parameters to make decisions that optimize the rewards received over time. An Internal brain uses the trained policy parameters to make decisions (and no longer adjusts the parameters in search of a better decision). The other types of brains do not directly involve training, but you might find them useful as part of a training project. See [Agent Brains](link).
  
### Observations and State

To make decisions, an agent must observe its environment to determine its current state. A state observation can take the following forms:

* **Continuous** — a feature vector consisting of an array of numbers. 
* **Discrete** — an index into a state table (typically only useful for the simplest of environments).
* **Camera** — one or more camera images.

When you use the **Continuous** or **Discrete** state space for an agent, implement the [CollectState()](link) method to create the feature vector or state index. When you use camera observations, you only need to identify which Unity Camera objects will provide images and the base Agent class handles the rest. You do not need to implement the `CollectState()` method.

#### Continuous State Space: Feature Vectors

For agents using a continuous state space, you create a feature vector to represent the agent's observation at each step of the simulation. The Brain class calls the `CollectState()` method of each of its agents. Your implementation of this function returns the feature vector observation as a `List<float>` object. 

The observation must include all the information an agent needs to accomplish its task. Without sufficient and relevant information, an agent may learn poorly or may not learn at all. A reasonable approach for determining what information should be included is to consider what you would need to calculate an analytical solution to the problem. 

For examples of various state observation functions, you can look at the [Examples](link) included in the ML AGents SDK.  For instance, the 3DBall example uses the rotation of the platform, the relative position of the ball, and the velocity of the ball as its state observation:

    public GameObject ball;

    public override List<float> CollectState()
    {
        List<float> state = new List<float>();
        state.Add(gameObject.transform.rotation.z);
        state.Add(gameObject.transform.rotation.x);
        state.Add((ball.transform.position.x - gameObject.transform.position.x));
        state.Add((ball.transform.position.y - gameObject.transform.position.y));
        state.Add((ball.transform.position.z - gameObject.transform.position.z));
        state.Add(ball.transform.GetComponent<Rigidbody>().velocity.x);
        state.Add(ball.transform.GetComponent<Rigidbody>().velocity.y);
        state.Add(ball.transform.GetComponent<Rigidbody>().velocity.z);
        return state;
    }

<!-- Note that the above values aren't normalized, which we recommend! -->

The feature vector must always contain the same number of elements and observations must always be in the same position within the list. If the number of observed entities in an environment can vary you can pad the feature vector with zeros for any missing entities in a specific observation or you can limit an agent's observations to a fixed subset. For example, instead of observing every enemy agent in an environment, you could only observe the closest five. 

When you set up an Agent's brain in the Unity Editor, set the following properties to use a continuous state-space feature vector:

[Screenshot of Brain Inspector]

**State Size** — The state size must match the length of your feature vector.
**State Space Type** — Set to **Continuous**.
**Brain Type** — Set to **External** during training; set to **Internal** to use the trained model.

The observation feature vector is a list of floating point numbers, which means you must convert any other data types to a float or a list of floats. 

Integers can be be added directly to the state vector, relying on implicit conversion in the `List.Add()` function. You must explicitly convert Boolean values to a number:

    state.Add(isTrueOrFalse ? 1 : 0);

For entities like positions and rotations, you can add their components to the feaure list individually.  For example:

    Vector3 speed = ball.transform.GetComponent<Rigidbody>().velocity;
    state.Add(speed.x);
    state.Add(speed.y);
    state.Add(speed.z);

Type enumerations should be encoded in the _one-hot_ style. That is, add an element to the feature vector for each element of enumeration. Set the element representing the observed member to one and set the rest to zero. For example, if your enumeration contains \[Sword, Shield, Bow\] and the agent observes a Bow, you would add the elements: 0, 0, 1 to the feature vector.

    [code example]

<!-- 
How to handle things like large numbers of strings or symbols? A very long one-hot vector? A single index into a table? 
Colors? Better to use a single color number or individual components?
-->

##### Normalization

For the best results when training, you should normalize the components of your feature vector to the range [-1, +1] or [0, 1]. When you normalize the values, the PPO neural network can often converge to a solution faster. Note that it isn't always necessary to normalize to these recommended ranges, but it is considered a best practice when using neural networks. The greater the variation in ranges between the components of your observation, the more likely that training will be affected.

To normalize a value to [0, 1], you can use the following formula:

    normalizedValue = (currentValue - minValue)/(maxValue - minValue)

Rotations should also be normalized. First, make sure that the components are in a standard range (like 0 to 360 or -180 to +180), and then normalize:

    Quaternion rotation = transform.rotation;
    Vector3 normalized = rotation.eulerAngles/180.0f - Vector3.one; //(-1..1)
    Vector3 normalized = rotation.eulerAngles/360.0f; //(0..1)
  
#### Camera Observations

Camera observations use rendered textures from one or more cameras in a scene. The brain vectorizes the textures and feeds them into a neural network.
 
 Agents using camera images can capture state of arbitrary complexity and are useful when the state is difficult to describe numerically. However, they are also typically less efficient and slower to train, and sometimes don't succeed at all.  
 
You can set up the cameras you use for observations in a variety of ways. 
 
#### Discrete State Space: Table Lookup

You can use the discrete state space when an agent only has a limited number of possible states and those states can be enumerated by a single number. For instance, the [Basic example environment](link) in the ML Agent SDK defines an agent with a discrete state space. The states of this agent are the integer steps between two linear goals. In the Basic example, the agent learns to move to the goal that provides the greatest reward.

More generally, the discrete state identifier could be an index into a table of the possible states. However, tables quickly become unwieldy as the environment becomes more complex. For example, even a simple game like [tic-tac-toe has 765 possible states](https://en.wikipedia.org/wiki/Game_complexity) (far more if you don't reduce the number of states by combining those that are rotations or reflections of each other).

To implement a discrete state observation, implement the [CollectState()](link) method of your Agent subclass and return a `List` containing a single number representing the state:

    public override List<float> CollectState()
    {
        List<float> state = new List<float>();
        state.Add(stateIndex); //stateIndex is the state identifier
        return state;
    }

### Rewards


### Actions

#### Continuous Action Space

#### Discrete Action Space

### Agent Brains

A Heuristic brain provides a hand-programmed decision making process, for example you could program a decision tree. You can use agents with heuristic brains as part of an environment, to help test an environment, to compare a trained agent to a hard-coded agent. You can even use a hybrid system where you use a heuristic brain in some states and a trained brain in others. 

A Player brain maps actions to keyboard keys so that a human can control an agent directly. Player brains are useful when testing an environment before training.

### Implementing an Agent Subclass
