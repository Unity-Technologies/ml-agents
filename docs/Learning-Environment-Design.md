# Reinforcement Learning in Unity

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
1. Calls the `OnEpisodeBegin()` function for each Agent in the scene.
1. Calls the  `CollectObservations(VectorSensor sensor)` function for each Agent in the scene.
1. Uses each Agent's Policy to decide on the Agent's next action.
1. Calls the `OnActionReceived()` function for each Agent in the scene, passing in
   the action chosen by the Agent's Policy.
1. Calls the Agent's `OnEpisodeBegin()` function if the Agent has reached its `Max
   Step` count or has otherwise marked itself as `EndEpisode()`.

To create a training environment, extend the Agent class to
implement the above methods whether you need to implement them or not depends on
your specific scenario.

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

When you create a training environment in Unity, you must set up the scene so
that it can be controlled by the external training process. Considerations
include:

* The training scene must start automatically when your Unity application is
  launched by the training process.
* The Academy must reset the scene to a valid starting point for each episode of
  training.
* A training episode must have a definite end — either using `Max Steps` or by
  each Agent ending its episode manually with `EndEpisode()`.
