# Reinforcement Learning in Unity

Reinforcement learning is an artificial intelligence technique that trains
_agents_ to perform tasks by rewarding desirable behavior. During reinforcement
learning, an agent explores its environment, observes the state of things, and,
based on those observations, takes an action. If the action leads to a better
state, the agent receives a positive reward. If it leads to a less desirable
state, then the agent receives no reward or a negative reward (punishment). As
the agent learns during training, it optimizes its decision making so that it
receives the maximum reward over time.

The ML-Agents toolkit uses a reinforcement learning technique called
[Proximal Policy Optimization (PPO)](https://blog.openai.com/openai-baselines-ppo/).
PPO uses a neural network to approximate the ideal function that maps an agent's
observations to the best action an agent can take in a given state. The
ML-Agents PPO algorithm is implemented in TensorFlow and runs in a separate
Python process (communicating with the running Unity application over a socket).

**Note:** if you aren't studying machine and reinforcement learning as a subject
and just want to train agents to accomplish tasks, you can treat PPO training as
a _black box_. There are a few training-related parameters to adjust inside
Unity as well as on the Python training side, but you do not need in-depth
knowledge of the algorithm itself to successfully create and train agents.
Step-by-step procedures for running the training process are provided in the
[Training section](Training-ML-Agents.md).

## The Simulation and Training Process

Training and simulation proceed in steps orchestrated by the ML-Agents Academy
class. The Academy works with Agent objects in the scene to step
through the simulation.

During training, the external Python training process communicates with the
Academy to run a series of episodes while it collects data and optimizes its
neural network model. When training is completed
successfully, you can add the trained model file to your Unity project for later
use.

The ML-Agents Academy class orchestrates the agent simulation loop as follows:

1. Calls your Academy's `OnEnvironmentReset` delegate.
2. Calls the `OnEpisodeBegin()` function for each Agent in the scene.
3. Calls the  `CollectObservations(VectorSensor sensor)` function for each Agent in the scene.
4. Uses each Agent's Policy to decide on the Agent's next action.
5. Calls the `OnActionReceived()` function for each Agent in the scene, passing in
   the action chosen by the Agent's Policy.
6. Calls the Agent's `OnEpisodeBegin()` function if the Agent has reached its `Max
   Step` count or has otherwise marked itself as `EndEpisode()`.

To create a training environment, extend the Agent class to
implement the above methods whether you need to implement them or not depends on
your specific scenario.

**Note:** The API used by the Python training process to communicate with
and control the Academy during training can be used for other purposes as well.
For example, you could use the API to use Unity as the simulation engine for
your own machine learning algorithms. See [Python API](Python-API.md) for more
information.

## Organizing the Unity Scene

To train and use the ML-Agents toolkit in a Unity scene, the scene as many Agent subclasses as you need.
Agent instances should be attached to the GameObject representing that Agent.

### Academy

The Academy is a singleton which orchestrates Agents and their decision making processes. Only
a single Academy exists at a time.

#### Academy resetting
To alter the environment at the start of each episode, add your method to the Academy's OnEnvironmentReset action.

```csharp
public class MySceneBehavior : MonoBehaviour
{
    public void Awake()
    {
        Academy.Instance.OnEnvironmentReset += EnvironmentReset;
    }

    void EnvironmentReset()
    {
        // Reset the scene here
    }
}
```

For example, you might want to reset an Agent to its starting
position or move a goal to a random position. An environment resets when the
`reset()` method is called on the Python `UnityEnvironment`.

When you reset an environment, consider the factors that should change so that
training is generalizable to different conditions. For example, if you were
training a maze-solving agent, you would probably want to change the maze itself
for each training episode. Otherwise, the agent would probably on learn to solve
one, particular maze, not mazes in general.

### Agent

The Agent class represents an actor in the scene that collects observations and
carries out actions. The Agent class is typically attached to the GameObject in
the scene that otherwise represents the actor — for example, to a player object
in a football game or a car object in a vehicle simulation. Every Agent must
have appropriate `Behavior Parameters`.

To create an Agent, extend the Agent class and implement the essential
`CollectObservations(VectorSensor sensor)` and `OnActionReceived()` methods:

* `CollectObservations(VectorSensor sensor)` — Collects the Agent's observation of its environment.
* `OnActionReceived()` — Carries out the action chosen by the Agent's Policy and
  assigns a reward to the current state.

Your implementations of these functions determine how the Behavior Parameters
assigned to this Agent must be set.

You must also determine how an Agent finishes its task or times out. You can
manually terminate an Agent episode in your `OnActionReceived()` function when the Agent
has finished (or irrevocably failed) its task by calling the `EndEpisode()` function.
You can also set the Agent's `Max Steps` property to a positive value and the
Agent will consider the episode over after it has taken that many steps. You can
use the `Agent.OnEpisodeBegin()` function to prepare the Agent to start again.

See [Agents](Learning-Environment-Design-Agents.md) for detailed information
about programming your own Agents.

## Environments

An _environment_ in the ML-Agents toolkit can be any scene built in Unity. The
Unity scene provides the environment in which agents observe, act, and learn.
How you set up the Unity scene to serve as a learning environment really depends
on your goal. You may be trying to solve a specific reinforcement learning
problem of limited scope, in which case you can use the same scene for both
training and for testing trained agents. Or, you may be training agents to
operate in a complex game or simulation. In this case, it might be more
efficient and practical to create a purpose-built training scene.

When you create a training environment in Unity, you must set up the scene so
that it can be controlled by the external training process. Considerations
include:

* The training scene must start automatically when your Unity application is
  launched by the training process.
* The Academy must reset the scene to a valid starting point for each episode of
  training.
* A training episode must have a definite end — either using `Max Steps` or by
  each Agent ending its episode manually with `EndEpisode()`.
