# Designing a Learning Environment

This page contains general advice on how to design your learning environment, in
addition to overviewing aspects of the ML-Agents Unity SDK that pertain to
setting up your scene and simulation as opposed to designing your agents within
the scene. We have a dedicated page on
[Designing Agents](Learning-Environment-Design-Agents.md) which includes how to
instrument observations, actions and rewards, define teams for multi-agent
scenarios and record agent demonstrations for imitation learning.

To help on-board to the entire set of functionality provided by the ML-Agents
Toolkit, we recommend exploring our [API documentation](API-Reference.md).
Additionally, our [example environments](Learning-Environment-Examples.md) are a
great resource as they provide sample usage of almost all of our features.

## The Simulation and Training Process

Training and simulation proceed in steps orchestrated by the ML-Agents Academy
class. The Academy works with Agent objects in the scene to step through the
simulation.

During training, the external Python training process communicates with the
Academy to run a series of episodes while it collects data and optimizes its
neural network model. When training is completed successfully, you can add the
trained model file to your Unity project for later use.

The ML-Agents Academy class orchestrates the agent simulation loop as follows:

1. Calls your Academy's `OnEnvironmentReset` delegate.
1. Calls the `OnEpisodeBegin()` function for each Agent in the scene.
1. Gathers information about the scene. This is done by calling the
  `CollectObservations(VectorSensor sensor)` function for each Agent in the
  scene, as well as updating their sensor and collecting the resulting
  observations.
1. Uses each Agent's Policy to decide on the Agent's next action.
1. Calls the `OnActionReceived()` function for each Agent in the scene, passing
   in the action chosen by the Agent's Policy.
1. Calls the Agent's `OnEpisodeBegin()` function if the Agent has reached its
   `Max Step` count or has otherwise marked itself as `EndEpisode()`.

To create a training environment, extend the Agent class to implement the above
methods whether you need to implement them or not depends on your specific
scenario.

## Organizing the Unity Scene

To train and use the ML-Agents Toolkit in a Unity scene, the scene as many Agent
subclasses as you need. Agent instances should be attached to the GameObject
representing that Agent.

### Academy

The Academy is a singleton which orchestrates Agents and their decision making
processes. Only a single Academy exists at a time.

#### Academy resetting

To alter the environment at the start of each episode, add your method to the
Academy's OnEnvironmentReset action.

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

For example, you might want to reset an Agent to its starting position or move a
goal to a random position. An environment resets when the `reset()` method is
called on the Python `UnityEnvironment`.

When you reset an environment, consider the factors that should change so that
training is generalizable to different conditions. For example, if you were
training a maze-solving agent, you would probably want to change the maze itself
for each training episode. Otherwise, the agent would probably on learn to solve
one, particular maze, not mazes in general.

### Multiple Areas

In many of the example environments, many copies of the training area are
instantiated in the scene. This generally speeds up training, allowing the
environment to gather many experiences in parallel. This can be achieved simply
by instantiating many Agents with the same Behavior Name. If possible, consider
designing your scene to support multiple areas.

Check out our example environments to see examples of multiple areas.
Additionally, the
[Making a New Learning Environment](Learning-Environment-Create-New.md#optional-multiple-training-areas-within-the-same-scene)
guide demonstrates this option.

## Environments

When you create a training environment in Unity, you must set up the scene so
that it can be controlled by the external training process. Considerations
include:

- The training scene must start automatically when your Unity application is
  launched by the training process.
- The Academy must reset the scene to a valid starting point for each episode of
  training.
- A training episode must have a definite end — either using `Max Steps` or by
  each Agent ending its episode manually with `EndEpisode()`.

## Environment Parameters

Curriculum learning and environment parameter randomization are two training
methods that control specific parameters in your environment. As such, it is
important to ensure that your environment parameters are updated at each step to
the correct values. To enable this, we expose a `EnvironmentParameters` C# class
that you can use to retrieve the values of the parameters defined in the
training configurations for both of those features. Please see our
[documentation](Training-ML-Agents.md#environment-parameters)
for curriculum learning and environment parameter randomization for details.

We recommend modifying the environment from the Agent's `OnEpisodeBegin()`
function by leveraging `Academy.Instance.EnvironmentParameters`. See the
WallJump example environment for a sample usage (specifically,
[WallJumpAgent.cs](../Project/Assets/ML-Agents/Examples/WallJump/Scripts/WallJumpAgent.cs)
).

## Agent

The Agent class represents an actor in the scene that collects observations and
carries out actions. The Agent class is typically attached to the GameObject in
the scene that otherwise represents the actor — for example, to a player object
in a football game or a car object in a vehicle simulation. Every Agent must
have appropriate `Behavior Parameters`.

Generally, when creating an Agent, you should extend the Agent class and implement
the `CollectObservations(VectorSensor sensor)` and `OnActionReceived()` methods:

- `CollectObservations(VectorSensor sensor)` — Collects the Agent's observation
  of its environment.
- `OnActionReceived()` — Carries out the action chosen by the Agent's Policy and
  assigns a reward to the current state.

Your implementations of these functions determine how the Behavior Parameters
assigned to this Agent must be set.

You must also determine how an Agent finishes its task or times out. You can
manually terminate an Agent episode in your `OnActionReceived()` function when
the Agent has finished (or irrevocably failed) its task by calling the
`EndEpisode()` function. You can also set the Agent's `Max Steps` property to a
positive value and the Agent will consider the episode over after it has taken
that many steps. You can use the `Agent.OnEpisodeBegin()` function to prepare
the Agent to start again.

See [Agents](Learning-Environment-Design-Agents.md) for detailed information
about programming your own Agents.

## Recording Statistics

We offer developers a mechanism to record statistics from within their Unity
environments. These statistics are aggregated and generated during the training
process. To record statistics, see the `StatsRecorder` C# class.

See the FoodCollector example environment for a sample usage (specifically,
[FoodCollectorSettings.cs](../Project/Assets/ML-Agents/Examples/FoodCollector/Scripts/FoodCollectorSettings.cs)
).
