# Creating Agents

An agent is an actor that can observe its environment and decide on the best course of action using those observations. Create agents in Unity by extending the Agent class. The most important aspects of creating agents that can successfully learn are the observations the agent collects and the reward you assign to estimate the value of the agent's current state toward accomplishing its tasks.

In the ML Agents framework, an agent passes its observations to its brain at each simulation step. The brain, then, makes a decision and passes the chosen action back to the agent. The agent code executes the action, for example, it moves the agent in one direction or another, and also calculates a reward based on the current state. In training, the reward is used to discover the optimal decision-making policy. (The reward is not used by already trained agents.)

The Brain class abstracts out the decision making logic from the agent itself so that you can use the same brain in multiple agents. 
How a brain makes its decisions depends on the type of brain it is. An **External** brain simply passes the observations from its agents to an external process and then passes the decisions made externally back to the agents. During training, the ML Agents [reinforcement learning](Reinforcement-Learning-in_Unity) algorithm adjusts its internal policy parameters to make decisions that optimize the rewards received over time. An Internal brain uses the trained policy parameters to make decisions (and no longer adjusts the parameters in search of a better decision). The other types of brains do not directly involve training, but you might find them useful as part of a training project. See [Agent Brains](link).
  
## Observations and State

To make decisions, an agent must observe its environment to determine its current state. A state observation can take the following forms:

* **Continuous** — a feature vector consisting of an array of numbers. 
* **Discrete** — an index into a state table (typically only useful for the simplest of environments).
* **Camera** — one or more camera images.

When you use the **Continuous** or **Discrete** state space for an agent, implement the `Agent.CollectState()` method to create the feature vector or state index. When you use camera observations, you only need to identify which Unity Camera objects will provide images and the base Agent class handles the rest. You do not need to implement the `CollectState()` method.

### Continuous State Space: Feature Vectors

For agents using a continuous state space, you create a feature vector to represent the agent's observation at each step of the simulation. The Brain class calls the `CollectState()` method of each of its agents. Your implementation of this function returns the feature vector observation as a `List<float>` object. 

The observation must include all the information an agent needs to accomplish its task. Without sufficient and relevant information, an agent may learn poorly or may not learn at all. A reasonable approach for determining what information should be included is to consider what you would need to calculate an analytical solution to the problem. 

For examples of various state observation functions, you can look at the [Examples](Example-Environments.md) included in the ML Agents SDK.  For instance, the 3DBall example uses the rotation of the platform, the relative position of the ball, and the velocity of the ball as its state observation. As an experiment, you can remove the velocity components from the observation and retrain the 3DBall agent. While it will learn to balance the ball reasonably well, the performance of the agent without using velocity is noticeably worse.

    public GameObject ball;
    
    private List<float> state = new List<float>();
    public override List<float> CollectState()
    {
        state.Clear();
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

Type enumerations should be encoded in the _one-hot_ style. That is, add an element to the feature vector for each element of enumeration, setting the element representing the observed member to one and set the rest to zero. For example, if your enumeration contains \[Sword, Shield, Bow\] and the agent observes that the current item is a Bow, you would add the elements: 0, 0, 1 to the feature vector. The following code example illustrates how to add 

    enum CarriedItems {Sword, Shield, Bow, LastItem}
    private List<float> state = new List<float>();
    public override List<float> CollectState()
    {
        state.Clear();
        for(int ci = 0; ci < (int)CarriedItems.LastItem; ci++){
            state.Add((int)currentItem == ci ? 1.0f : 0.0f);            
        }
        return state;
    }


<!-- 
How to handle things like large numbers of words or symbols? Should you use a very long one-hot vector? Or a single index into a table? 
Colors? Better to use a single color number or individual components?
-->

#### Normalization

For the best results when training, you should normalize the components of your feature vector to the range [-1, +1] or [0, 1]. When you normalize the values, the PPO neural network can often converge to a solution faster. Note that it isn't always necessary to normalize to these recommended ranges, but it is considered a best practice when using neural networks. The greater the variation in ranges between the components of your observation, the more likely that training will be affected.

To normalize a value to [0, 1], you can use the following formula:

    normalizedValue = (currentValue - minValue)/(maxValue - minValue)

Rotations and angles should also be normalized. For angles between 0 and 360 degrees, you can use the following formulas:

    Quaternion rotation = transform.rotation;
    Vector3 normalized = rotation.eulerAngles/180.0f - Vector3.one; //[-1,1]
    Vector3 normalized = rotation.eulerAngles/360.0f; //[0,1]
  
 For angles that can be outside the range [0,360], you can either reduce the angle, or, if the number of turns is significant, increase the maximum value used in your normalization formula.
 
### Camera Observations

Camera observations use rendered textures from one or more cameras in a scene. The brain vectorizes the textures and feeds them into a neural network.
 
 Agents using camera images can capture state of arbitrary complexity and are useful when the state is difficult to describe numerically. However, they are also typically less efficient and slower to train, and sometimes don't succeed at all.  
  
### Discrete State Space: Table Lookup

You can use the discrete state space when an agent only has a limited number of possible states and those states can be enumerated by a single number. For instance, the [Basic example environment](link) in the ML Agent SDK defines an agent with a discrete state space. The states of this agent are the integer steps between two linear goals. In the Basic example, the agent learns to move to the goal that provides the greatest reward.

More generally, the discrete state identifier could be an index into a table of the possible states. However, tables quickly become unwieldy as the environment becomes more complex. For example, even a simple game like [tic-tac-toe has 765 possible states](https://en.wikipedia.org/wiki/Game_complexity) (far more if you don't reduce the number of states by combining those that are rotations or reflections of each other).

To implement a discrete state observation, implement the `CollectState()` method of your Agent subclass and return a `List` containing a single number representing the state:

    private List<float> state = new List<float>();
    public override List<float> CollectState()
    {
        state[0] = stateIndex; //stateIndex is the state identifier
        return state;
    }

## Actions

An action is an instruction from the brain that the agent carries out. The action is passed to the agent as a parameter when the Academy invokes the agent's `AgentStep()` function. When you specify that the action space is **Continuous**, the action parameter passed to the agent is an array of control signals with length equal to the `Action Size` property.  When you specify a **Discrete** action space, the action parameter is an array containing only a single value, which is an index into your list or table of commands. In the **Discrete** action space, the `Action Size` is the number of elements in your action table. Set the `Action Space` and `Action Size` properties on the Brain object assigned to the agent (using the Unity Editor Inspector window). 

Neither the Brain nor the training algorithm know anything about what the action values themselves mean. The training algorithm simply tries different values for the action list and observes the affect on the accumulated rewards over time and many training episodes. Thus, the only place actions are defined for an agent is in the `AgentStep()` function. You simply specify the type of action space, and, for the coninuous action space, the number of values, and then apply the received values appropriately (and consistently) in `ActionStep()`.

For example, if you designed an agent to move in two dimensions, you could use either continuous or the discrete actions. In the continuous case, you would set the action size to two (one for each dimension), and the agent's brain would create an action with two floating point values. In the discrete case, you would set the action size to four (one for each direction), and the brain would create an action array containing a single element with a value ranging from zero to four.  

Note that when you are programming actions for an agent, it is often helpful to test your action logic using a **Player** brain, which lets you map keyboard commands to actions. See [Agent Brains](link).

The [3DBall and Area example projects](Example-Environments.md) are set up to use either the continuous or the discrete action spaces. 

### Continuous Action Space

When an agent uses a brain set to the **Continuous** action space, the action parameter passed to the agent's `AgentStep()` function is an array with length equal to the Brain object's `Action Size` property value.  The individual values in the array have whatever meanings that you ascribe to them. If you assign an element in the array as the speed of an agent, for example, the training process learns to control the speed of the agent though this parameter. 

The [Reacher example](Example-Environments.md) defines a continuous action space with four control values. 

[screenshot of reacher]

These control values are applied as torques to the bodies making up the arm :

    public override void AgentStep(float[] act)
    {
        float torque_x = Mathf.Clamp(act[0], -1, 1) * 100f;
        float torque_z = Mathf.Clamp(act[1], -1, 1) * 100f;
        rbA.AddTorque(new Vector3(torque_x, 0f, torque_z));
    
        torque_x = Mathf.Clamp(act[2], -1, 1) * 100f;
        torque_z = Mathf.Clamp(act[3], -1, 1) * 100f;
        rbB.AddTorque(new Vector3(torque_x, 0f, torque_z));
    }

You should clamp continuous action values to a reasonable value (typically [-1,1]) to avoid introducing instability while training the agent with the PPO algorithm. As shown above, you can scale the control values as needed after clamping them. 
 
### Discrete Action Space

When an agent uses a brain set to the **Discrete** action space, the action parameter passed to the agent's `AgentStep()` function is an array containing a single element. The value is the index of the action to in your table or list of actions. With the discrete action space, `Action Size` represents the number of actions in your action table.

The [Area example](Example-Environments.md) defines five actions for the discrete action space: a jump action and one action for each cardinal direction:

    // Get the action index
    int movement = Mathf.FloorToInt(act[0]); 
    // Look up the index in the action list:
    if (movement == 1) { directionX = -1; }
    if (movement == 2) { directionX = 1; }
    if (movement == 3) { directionZ = -1; }
    if (movement == 4) { directionZ = 1; }
    if (movement == 5 && GetComponent<Rigidbody>().velocity.y <= 0) { directionY = 1; }
    
    // Apply the action results to move the agent
    gameObject.GetComponent<Rigidbody>().AddForce(new Vector3(directionX * 40f, directionY * 300f, directionZ * 40f));

Note that the above code example is a simplified extract from the AreaAgent class, which provides alternate implementations for both the discrete and the continuous action spaces.

## Rewards

A reward is a signal that the agent has done something right. The PPO reinforcement learning algorithm works by optimizing the choices an agent makes such that the agent earns the highest cumulative reward over time. The better your reward mechanism, the better your agent will learn.

Perhaps the best advice is to start simple and only add complexity as needed. In general, you should reward results rather than actions you think will lead to the desired results. To help develop your rewards, you can use the Monitor class to display the cumulative reward recieved by an agent. You can even use a Player brain to control the agent while watching how it accumulates rewards.

Allocate rewards to an agent by setting the agent's `reward` property in the `AgentStep()` function. The reward assigned in any step should be in the range [-1,1].  Values outside this range can lead to unstable training. The `reward` value is reset to zero at every step. 

**Examples**

You can examine the `AgentStep()` functions defined in the [Examples](link) to see how those projects allocate rewards.

The `GridAgent` class in the [GridWorld example](Example-Environments.md) uses a very simple reward system:

    Collider[] hitObjects = Physics.OverlapBox(trueAgent.transform.position, new Vector3(0.3f, 0.3f, 0.3f));
    if (hitObjects.Where(col => col.gameObject.tag == "goal").ToArray().Length == 1)
    {
        reward = 1f;
        done = true;
    }
    if (hitObjects.Where(col => col.gameObject.tag == "pit").ToArray().Length == 1)
    {
        reward = -1f;
        done = true;
    }

The agent receives a positive reward when it reaches the goal and a negative reward when it falls into the pit. Otherwise, it gets no rewards. This is an example of a _sparse_ reward system. The agent must explore a lot to find the infrequent reward.

In contrast, the `AreaAgent` in the [Area example](Example-Environments.md) gets a small negative reward every step. In order to get the maximum reward, the agent must finish its task of reaching the goal square as quickly as possible:

	reward = -0.005f;
    MoveAgent(act);
    
	if (gameObject.transform.position.y < 0.0f || Mathf.Abs(gameObject.transform.position.x - area.transform.position.x) > 8f || 
        Mathf.Abs(gameObject.transform.position.z + 5 - area.transform.position.z) > 8)
	{
		done = true;
		reward = -1f;
	}

The agent also gets a larger negative penalty if it falls off the playing surface.

The `Ball3DAgent` in the [3DBall](Example-Environments.md) takes a similar approach, but allocates a small positive reward as long as the agent balances the ball. The agent can maximize its rewards by keeping the ball on the platform:

    if (done == false)
    {
        reward = 0.1f;
    }
    
    //When ball falls mark agent as done and give a negative penalty
    if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
        Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
        Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
    {
        done = true;
        reward = -1f;
    }

The `Ball3DAgent` also assigns a negative penalty when the ball falls off the platfrom.

