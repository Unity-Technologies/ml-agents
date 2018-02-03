# Agents

An agent is an actor that can observe its environment and decide on the best course of action using those observations. You create agents in Unity by extending the [Agent](link) class. The most important aspects of creating agents that can successfully learn tasks using reinforcement learning are the observations the agent collects and the reward you assign to estimate the value of the agent's current state in accomplishing its tasks.

<!-- We could discuss more about the arbitrary line between actor and environment here, but it can wait until we have examples of multi-agent entities (e.g. something like a executive brain setting goals for a "robot" and sub-brains moving the robot limbs to accomplish these goals). -->

In the ML Agents framework, an agent passes its observations to its brain at each simulation step. The brain, then, makes a decision and passes the chosen action back to the agent. The agent code executes the action, for example, it moves the agent in one direction or another, and also calculates a reward based on the current state. In training, the reward is used to discover the optimal decision-making policy. (The reward is not used by already trained agents.)

The Brain class abstracts out the decision making logic from the agent itself so that you can use the same brain in multiple agents. 
How a brain makes its decisions depends on the type of brain it is. In training, an External brain adjusts its internal policy parameters to optimize the rewards received over time. An Internal brain uses the trained policy parameters to make decisions (and no longer adjusts the parameters in search of a better decision). The other types of brains do not directly involve training, but you might find them useful as part of a training project. See [Agent Brains](link).

  
### State Observations

To make decisions, an agent must observe its environment to determine its current state. A state observation can take the following forms:

* Continuous: a feature vector consisting of an array of numbers. 
* Camera observation: one or more camera images.
* Discrete: an index into a state table (typically only useful for the simplest of environments).

The main choice to make when creating an agent is whether to use a feature vector or a camera image as the agent's observation. Discrete state observations can only be used when the state space can be described by a single number, such as an index into a table of states, which is typically only practical for very simple cases.

Agents using feature vectors are typically more efficient and faster to train than those using camera observations, but they also require more detailed knowledge of the internal workings of the simulation or game. Conversely, agents using camera images can capture state of arbitrary complexity and are useful when the state is difficult to describe numerically. However, agents using camera images typically require more computation per decision, are slower to train, and sometimes don't succeed at all. If an agent using a feature vector fails to learn, you can often add more or different data to its observation to improve your results.  

When you create an Agent subclass, implement the [CollectState()](link) method to create the feature vector. The Brain object assigned to an agent uses the feature vector to decide which of the possible actions to take. When you use camera observations, you only need to identify which Unity Camera objects will provide images and the base Agent class handles the rest. You do not need to implement the `CollectState()` method.

#### Continuous State -- Feature Vectors

For agents using a continuous state space, you create a feature vector to represent the agent's observation at each step of the simulation. The Brain class calls the `CollectState()` method of each of its agents. Your implementation of this function returns the feature vector observation as a `List<float>` object. 

The observation must include all the information an agent needs to accomplish its task. Without sufficient and relevant information, an agent will never learn. For instance, the 3DBall example uses the rotation of the platform, the relative position of the ball, and the velocity of the ball as its state observation. 

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

The feature vector must always contain the same number of elements and observations must always be in the same position within the list. If the number of entities under observation can vary in the environment See [Variable Length Observations](link).

When you set up an Agent's brain in the Unity Editor, set the following properties to use a continuous state-space feature vector:

**State Size** — The state size must match the length of your feature vector.
**State Space Type** — Set to **Continuous**.
**Brain Type** — Set to **External** during training; set to **Internal** to use the trained model.

[Screenshot of Brain Inspector]

##### Normalization

For the best results when training, you should normalize the components of your feature vector to the range [-1, +1] or [0, 1]. When you normalize the values, the PPO neural network can often converge to a solution faster. Note that it isn't always necessary to normalize to the recommended ranges, but it is considered a best practice when using neural networks. The greater the variation in ranges between the components of your observation, the more likely that traing will be affected.

To normalize a value to [0, 1], you can use the following formula:

    normalizedValue = (currentValue - minValue)/(maxValue - minValue)

Rotations should also be normalized. First, make sure that the components are in a standard range (like 0 to 360 or -180 to +180), and then normalize:

    Quaternion rotation = transform.rotation;
    Vector3 normalized = rotation.eulerAngles/180.0f - Vector3.one; //(-1..1)
    Vector3 normalized = rotation.eulerAngles/360.0f; //(0..1)



##### Data Format of a Feature Vector

The observation feature vector returned by `CollectState()` is of type `List<float>` -- in other words, a list of floating point numbers. This means you must convert any other data types to a float or a list of floats. 

Integers can be be added directly to the state vector, relying on implicit conversion in the `List.Add()` function. You must explicitly convert Boolean values to a number:

    state.Add(isTrueOrFalse ? 1 : 0);

For entities like positions and rotations, you can add their components to the feaure list individually.  For example:

    Vector3 speed = ball.transform.GetComponent<Rigidbody>().velocity;
    state.Add(speed.x);
    state.Add(speed.y);
    state.Add(speed.z);

Type enumerations should be encoded in the _one-hot_ style. That is, add an element to the feature vector for each element of enumeration. Set the element representing the observed member to one and set the rest to zero. For example, if your enumeration contains \[Sword, Shield, Bow\] and the agent observes a Bow, you would add the elements: 0, 0, 1 to the feature vector.


<!-- 
How to handle things like large numbers of strings or symbols? A very long one-hot vector? A single index into a table? 
Colors? Better to use a single color number or individual components?
-->

##### Variable Length Observations

When an agent can encounter a large and variable number of entities in its environment, it can be more effective to reduce the scope of an agent's observation rather than include a large, sparse array in the feature vector. 

This can have the added benefit of making the agent more generalizable as well.
 
  
#### Camera Observations

Camera observations use rendered textures from one or more cameras in a scene and feed them into a neural network.
 
#### Discrete State -- Table Lookup

You can use the discrete state space when an agent only has a limited number of possible states and those states can be enumerated by a single number. For instance, the [Basic example environment](link) in the ML Agent SDK defines an agent with a discrete state space. The states of this agent are the integer steps between two linear goals. In the Basic example, the agent learns to move to the goal that provides the greatest reward.

More generally, the discrete state identifier could be an index into a table of the possible states. However, tables quickly become unwieldy as the environment becomes more complex. For example, even a simple game like [tic-tac-toe has 765 possible states](https://en.wikipedia.org/wiki/Game_complexity) (more if you don't reduce the state space by combining states that are rotations or reflections of each other).

To implement a discrete state observation, implement the [CollectState()](link) method of your Agent subclass and return a `List` containing a single number representing the state:

    public override List<float> CollectState()
    {
        List<float> state = new List<float>();
        state.Add(stateIndex); //stateIndex is the state identifier
        return state;
    }

### Rewards


### Actions

### Agent Brains

A Heuristic brain provides a hand-programmed decision making process, for example you could program a decision tree. You can use agents with heuristic brains as part of an environment, to help test an environment, to compare a trained agent to a hard-coded agent. You can even use a hybrid system where you use a heuristic brain in some states and a trained brain in others. A Player brain maps actions to keyboard keys so that a human can control an agent directly. Player brains are useful when testing an environment before training.

### Implementing an Agent Subclass
