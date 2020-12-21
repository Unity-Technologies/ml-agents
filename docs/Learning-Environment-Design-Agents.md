# Agents

**Table of Contents:**

- [Decisions](#decisions)
- [Observations and Sensors](#observations-and-sensors)
  - [Generating Observations](#generating-observations)
    - [Agent.CollectObservations()](#agentcollectobservations)
    - [Observable Fields and Properties](#observable-fields-and-properties)
    - [ISensor interface and SensorComponents](#isensor-interface-and-sensorcomponents)
  - [Vector Observations](#vector-observations)
    - [One-hot encoding categorical information](#one-hot-encoding-categorical-information)
    - [Normalization](#normalization)
    - [Stacking](#stacking)
    - [Vector Observation Summary & Best Practices](#vector-observation-summary--best-practices)
  - [Visual Observations](#visual-observations)
    - [Visual Observation Summary & Best Practices](#visual-observation-summary--best-practices)
  - [Raycast Observations](#raycast-observations)
    - [RayCast Observation Summary & Best Practices](#raycast-observation-summary--best-practices)
- [Actions](#actions)
  - [Continuous Actions](#continuous-actions)
  - [Discrete Actions](#discrete-actions)
    - [Masking Discrete Actions](#masking-discrete-actions)
  - [Actions Summary & Best Practices](#actions-summary--best-practices)
- [Rewards](#rewards)
  - [Examples](#examples)
  - [Rewards Summary & Best Practices](#rewards-summary--best-practices)
- [Agent Properties](#agent-properties)
- [Destroying an Agent](#destroying-an-agent)
- [Defining Teams for Multi-agent Scenarios](#defining-teams-for-multi-agent-scenarios)
- [Recording Demonstrations](#recording-demonstrations)

An agent is an entity that can observe its environment, decide on the best
course of action using those observations, and execute those actions within its
environment. Agents can be created in Unity by extending the `Agent` class. The
most important aspects of creating agents that can successfully learn are the
observations the agent collects, and the reward you assign to estimate the value
of the agent's current state toward accomplishing its tasks.

An Agent passes its observations to its Policy. The Policy then makes a decision
and passes the chosen action back to the agent. Your agent code must execute the
action, for example, move the agent in one direction or another. In order to
[train an agent using reinforcement learning](Learning-Environment-Design.md),
your agent must calculate a reward value at each action. The reward is used to
discover the optimal decision-making policy.

The `Policy` class abstracts out the decision making logic from the Agent itself
so that you can use the same Policy in multiple Agents. How a Policy makes its
decisions depends on the `Behavior Parameters` associated with the agent. If you
set `Behavior Type` to `Heuristic Only`, the Agent will use its `Heuristic()`
method to make decisions which can allow you to control the Agent manually or
write your own Policy. If the Agent has a `Model` file, its Policy will use the
neural network `Model` to take decisions.

When you create an Agent, you should usually extend the base Agent class. This
includes implementing the following methods:

- `Agent.OnEpisodeBegin()` — Called at the beginning of an Agent's episode,
  including at the beginning of the simulation.
- `Agent.CollectObservations(VectorSensor sensor)` — Called every step that the Agent
  requests a decision. This is one possible way for collecting the Agent's
  observations of the environment; see [Generating Observations](#generating-observations)
  below for more options.
- `Agent.OnActionReceived()` — Called every time the Agent receives an action to
  take. Receives the action chosen by the Agent. It is also common to assign a
  reward in this method.
- `Agent.Heuristic()` - When the `Behavior Type` is set to `Heuristic Only` in
  the Behavior Parameters of the Agent, the Agent will use the `Heuristic()`
  method to generate the actions of the Agent. As such, the `Heuristic()` method
  writes to the array of floats provided to the Heuristic method as argument.
  __Note__: Do not create a new float array of action in the `Heuristic()` method,
  as this will prevent writing floats to the original action array.

As a concrete example, here is how the Ball3DAgent class implements these methods:

- `Agent.OnEpisodeBegin()` — Resets the agent cube and ball to their starting
  positions. The function randomizes the reset values so that the training
  generalizes to more than a specific starting position and agent cube
  orientation.
- `Agent.CollectObservations(VectorSensor sensor)` — Adds information about the
  orientation of the agent cube, the ball velocity, and the relative position
  between the ball and the cube. Since the  `CollectObservations()`
  method calls `VectorSensor.AddObservation()` such that vector size adds up to 8,
  the Behavior Parameters of the Agent are set with vector observation space
  with a state size of 8.
- `Agent.OnActionReceived()` — The action results
  in a small change in the agent cube's rotation at each step. In this example,
  an Agent receives a small positive reward for each step it keeps the ball on the
  agent cube's head and a larger, negative reward for dropping the ball. An
  Agent's episode is also ended when it drops the ball so that it will reset
  with a new ball for the next simulation step.
- `Agent.Heuristic()` - Converts the keyboard inputs into actions.

## Decisions

The observation-decision-action-reward cycle repeats each time the Agent request
a decision. Agents will request a decision when `Agent.RequestDecision()` is
called. If you need the Agent to request decisions on its own at regular
intervals, add a `Decision Requester` component to the Agent's GameObject.
Making decisions at regular step intervals is generally most appropriate for
physics-based simulations. For example, an agent in a robotic simulator that
must provide fine-control of joint torques should make its decisions every step
of the simulation. On the other hand, an agent that only needs to make decisions
when certain game or simulation events occur, such as in a turn-based game,
should call `Agent.RequestDecision()` manually.

## Observations and Sensors
In order for an agent to learn, the observations should include all the
information an agent needs to accomplish its task. Without sufficient and
relevant information, an agent may learn poorly or may not learn at all. A
reasonable approach for determining what information should be included is to
consider what you would need to calculate an analytical solution to the problem,
or what you would expect a human to be able to use to solve the problem.

### Generating Observations
ML-Agents provides multiple ways for an Agent to make observations:
  1. Overriding the `Agent.CollectObservations()` method and passing the
    observations to the provided `VectorSensor`.
  1. Adding the `[Observable]` attribute to fields and properties on the Agent.
  1. Implementing the `ISensor` interface, using a `SensorComponent` attached to
  the Agent to create the `ISensor`.

#### Agent.CollectObservations()
Agent.CollectObservations() is best used for aspects of the environment which are
numerical and non-visual. The Policy class calls the
`CollectObservations(VectorSensor sensor)` method of each Agent. Your
implementation of this function must call `VectorSensor.AddObservation` to add
vector observations.

The `VectorSensor.AddObservation` method provides a number of overloads for
adding common types of data to your observation vector. You can add Integers and
booleans directly to the observation vector, as well as some common Unity data
types such as `Vector2`, `Vector3`, and `Quaternion`.

For examples of various state observation functions, you can look at the
[example environments](Learning-Environment-Examples.md) included in the
ML-Agents SDK. For instance, the 3DBall example uses the rotation of the
platform, the relative position of the ball, and the velocity of the ball as its
state observation.

```csharp
public GameObject ball;

public override void CollectObservations(VectorSensor sensor)
{
    // Orientation of the cube (2 floats)
    sensor.AddObservation(gameObject.transform.rotation.z);
    sensor.AddObservation(gameObject.transform.rotation.x);
    // Relative position of the ball to the cube (3 floats)
    sensor.AddObservation(ball.transform.position - gameObject.transform.position);
    // Velocity of the ball (3 floats)
    sensor.AddObservation(m_BallRb.velocity);
    // 8 floats total
}
```

As an experiment, you can remove the velocity components from
the observation and retrain the 3DBall agent. While it will learn to balance the
ball reasonably well, the performance of the agent without using velocity is
noticeably worse.

The observations passed to `VectorSensor.AddObservation()` must always contain
the same number of elements must always be in the same order. If the number
of observed entities in an environment can vary, you can pad the calls
with zeros for any missing entities in a specific observation, or you can limit
an agent's observations to a fixed subset. For example, instead of observing
every enemy in an environment, you could only observe the closest five.

Additionally, when you set up an Agent's `Behavior Parameters` in the Unity
Editor, you must set the **Vector Observations > Space Size**
to equal the number of floats that are written by `CollectObservations()`.

#### Observable Fields and Properties
Another approach is to define the relevant observations as fields or properties
on your Agent class, and annotate them with an `ObservableAttribute`. For
example, in the 3DBall example above, the rigid body velocity could be observed
by adding a property to the Agent:
```csharp
using Unity.MLAgents.Sensors.Reflection;

public class Ball3DAgent : Agent {

    [Observable]
    public Vector3 RigidBodyVelocity
    {
        get { return m_BallRb.velocity;  }
    }
}
```
`ObservableAttribute` currently supports most basic types (e.g. floats, ints,
bools), as well as `Vector2`, `Vector3`, `Vector4`, `Quaternion`, and enums.

The behavior of `ObservableAttribute`s are controlled by the "Observable Attribute
Handling" in the Agent's `Behavior Parameters`. The possible values for this are:
 * **Ignore** (default) - All ObservableAttributes on the Agent will be ignored.
  If there are no ObservableAttributes on the Agent, this will result in the
  fastest  initialization time.
 * **Exclude Inherited** - Only members on the declared class will be examined;
  members that are inherited are ignored. This is a reasonable tradeoff between
  performance and flexibility.
 * **Examine All** All members on the class will be examined. This can lead to
  slower startup times.

"Exclude Inherited" is generally sufficient, but if your Agent inherits from
another Agent implementation that has Observable members, you will need to use
"Examine All".

Internally, ObservableAttribute uses reflection to determine which members of
the Agent have ObservableAttributes, and also uses reflection to access the
fields or invoke the properties at runtime. This may be slower than using
CollectObservations or an ISensor, although this might not be enough to
noticeably affect performance.

**NOTE**: you do not need to adjust the Space Size in the Agent's
`Behavior Parameters` when you add `[Observable]` fields or properties to an
Agent, since their size can be computed before they are used.

#### ISensor interface and SensorComponents
The `ISensor` interface is generally intended for advanced users. The `Write()`
method is used to actually generate the observation, but some other methods
such as returning the shape of the observations must also be implemented.

The `SensorComponent` abstract class is used to create the actual `ISensor` at
runtime. It must be attached to the same `GameObject` as the `Agent`, or to a
child `GameObject`.

There are several SensorComponents provided in the API:
- `CameraSensorComponent` - Allows image from `Camera` to be used as
  observation.
- `RenderTextureSensorComponent` - Allows content of `RenderTexture` to be used
  as observation.
- `RayPerceptionSensorComponent` - Allows information from set of ray-casts to
  be used as observation.

**NOTE**: you do not need to adjust the Space Size in the Agent's
`Behavior Parameters` when using an ISensor SensorComponents.

Internally, both `Agent.CollectObservations` and `[Observable]` attribute use an
ISensors to write observations, although this is mostly abstracted from the user.

### Vector Observations
Both `Agent.CollectObservations()` and `ObservableAttribute`s produce vector
observations, which are represented at lists of `float`s. `ISensor`s can
produce both vector observations and visual observations, which are
multi-dimensional arrays of floats.

Below are some additional considerations when dealing with vector observations:

#### One-hot encoding categorical information

Type enumerations should be encoded in the _one-hot_ style. That is, add an
element to the feature vector for each element of enumeration, setting the
element representing the observed member to one and set the rest to zero. For
example, if your enumeration contains \[Sword, Shield, Bow\] and the agent
observes that the current item is a Bow, you would add the elements: 0, 0, 1 to
the feature vector. The following code example illustrates how to add.

```csharp
enum ItemType { Sword, Shield, Bow, LastItem }
public override void CollectObservations(VectorSensor sensor)
{
    for (int ci = 0; ci < (int)ItemType.LastItem; ci++)
    {
        sensor.AddObservation((int)currentItem == ci ? 1.0f : 0.0f);
    }
}
```

`VectorSensor` also provides a two-argument function `AddOneHotObservation()` as
a shortcut for _one-hot_ style observations. The following example is identical
to the previous one.

```csharp
enum ItemType { Sword, Shield, Bow, LastItem }
const int NUM_ITEM_TYPES = (int)ItemType.LastItem;

public override void CollectObservations(VectorSensor sensor)
{
    // The first argument is the selection index; the second is the
    // number of possibilities
    sensor.AddOneHotObservation((int)currentItem, NUM_ITEM_TYPES);
}
```

`ObservableAttribute` has built-in support for enums. Note that you don't need
the `LastItem` placeholder in this case:
```csharp
enum ItemType { Sword, Shield, Bow }

public class HeroAgent : Agent
{
    [Observable]
    ItemType m_CurrentItem;
}
```

#### Normalization

For the best results when training, you should normalize the components of your
feature vector to the range [-1, +1] or [0, 1]. When you normalize the values,
the PPO neural network can often converge to a solution faster. Note that it
isn't always necessary to normalize to these recommended ranges, but it is
considered a best practice when using neural networks. The greater the variation
in ranges between the components of your observation, the more likely that
training will be affected.

To normalize a value to [0, 1], you can use the following formula:

```csharp
normalizedValue = (currentValue - minValue)/(maxValue - minValue)
```

:warning: For vectors, you should apply the above formula to each component (x,
y, and z). Note that this is _not_ the same as using the `Vector3.normalized`
property or `Vector3.Normalize()` method in Unity (and similar for `Vector2`).

Rotations and angles should also be normalized. For angles between 0 and 360
degrees, you can use the following formulas:

```csharp
Quaternion rotation = transform.rotation;
Vector3 normalized = rotation.eulerAngles / 180.0f - Vector3.one;  // [-1,1]
Vector3 normalized = rotation.eulerAngles / 360.0f;  // [0,1]
```

For angles that can be outside the range [0,360], you can either reduce the
angle, or, if the number of turns is significant, increase the maximum value
used in your normalization formula.

#### Stacking
Stacking refers to repeating observations from previous steps as part of a
larger observation. For example, consider an Agent that generates these
observations in four steps
```
step 1: [0.1]
step 2: [0.2]
step 3: [0.3]
step 4: [0.4]
```

If we use a stack size of 3, the observations would instead be:
```csharp
step 1: [0.1, 0.0, 0.0]
step 2: [0.2, 0.1, 0.0]
step 3: [0.3, 0.2, 0.1]
step 4: [0.4, 0.3, 0.2]
```
(The observations are padded with zeroes for the first `stackSize-1` steps).
This is a simple way to give an Agent limited "memory" without the complexity
of adding a recurrent neural network (RNN).

The steps for enabling stacking depends on how you generate observations:
* For Agent.CollectObservations(), set "Stacked Vectors" on the Agent's
  `Behavior Parameters` to a value greater than 1.
* For ObservableAttribute, set the `numStackedObservations` parameter in the
  constructor, e.g. `[Observable(numStackedObservations: 2)]`.
* For `ISensor`s, wrap them in a `StackingSensor` (which is also an `ISensor`).
  Generally, this should happen in the `CreateSensor()` method of your
  `SensorComponent`.

Note that stacking currently only supports for vector observations; stacking
for visual observations is not supported.

#### Vector Observation Summary & Best Practices

- Vector Observations should include all variables relevant for allowing the
  agent to take the optimally informed decision, and ideally no extraneous
  information.
- In cases where Vector Observations need to be remembered or compared over
  time, either an RNN should be used in the model, or the `Stacked Vectors`
  value in the agent GameObject's `Behavior Parameters` should be changed.
- Categorical variables such as type of object (Sword, Shield, Bow) should be
  encoded in one-hot fashion (i.e. `3` -> `0, 0, 1`). This can be done
  automatically using the `AddOneHotObservation()` method of the `VectorSensor`,
  or using `[Observable]` on an enum field or property of the Agent.
- In general, all inputs should be normalized to be in the range 0 to +1 (or -1
  to 1). For example, the `x` position information of an agent where the maximum
  possible value is `maxValue` should be recorded as
  `VectorSensor.AddObservation(transform.position.x / maxValue);` rather than
  `VectorSensor.AddObservation(transform.position.x);`.
- Positional information of relevant GameObjects should be encoded in relative
  coordinates wherever possible. This is often relative to the agent position.

### Visual Observations

Visual observations are generally provided to agent via either a `CameraSensor`
or `RenderTextureSensor`. These collect image information and transforms it into
a 3D Tensor which can be fed into the convolutional neural network (CNN) of the
agent policy. For more information on CNNs, see
[this guide](http://cs231n.github.io/convolutional-networks/). This allows
agents to learn from spatial regularities in the observation images. It is
possible to use visual and vector observations with the same agent.

Agents using visual observations can capture state of arbitrary complexity and
are useful when the state is difficult to describe numerically. However, they
are also typically less efficient and slower to train, and sometimes don't
succeed at all as compared to vector observations. As such, they should only be
used when it is not possible to properly define the problem using vector or
ray-cast observations.

Visual observations can be derived from Cameras or RenderTextures within your
scene. To add a visual observation to an Agent, add either a Camera Sensor
Component or RenderTextures Sensor Component to the Agent. Then drag the camera
or render texture you want to add to the `Camera` or `RenderTexture` field. You
can have more than one camera or render texture and even use a combination of
both attached to an Agent. For each visual observation, set the width and height
of the image (in pixels) and whether or not the observation is color or
grayscale.

![Agent Camera](images/visual-observation.png)

or

![Agent RenderTexture](images/visual-observation-rendertexture.png)

Each Agent that uses the same Policy must have the same number of visual
observations, and they must all have the same resolutions (including whether or
not they are grayscale). Additionally, each Sensor Component on an Agent must
have a unique name so that they can be sorted deterministically (the name must
be unique for that Agent, but multiple Agents can have a Sensor Component with
the same name).

When using `RenderTexture` visual observations, a handy feature for debugging is
adding a `Canvas`, then adding a `Raw Image` with it's texture set to the
Agent's `RenderTexture`. This will render the agent observation on the game
screen.

![RenderTexture with Raw Image](images/visual-observation-rawimage.png)

The [GridWorld environment](Learning-Environment-Examples.md#gridworld) is an
example on how to use a RenderTexture for both debugging and observation. Note
that in this example, a Camera is rendered to a RenderTexture, which is then
used for observations and debugging. To update the RenderTexture, the Camera
must be asked to render every time a decision is requested within the game code.
When using Cameras as observations directly, this is done automatically by the
Agent.

![Agent RenderTexture Debug](images/gridworld.png)

#### Visual Observation Summary & Best Practices

- To collect visual observations, attach `CameraSensor` or `RenderTextureSensor`
  components to the agent GameObject.
- Visual observations should generally be used unless vector observations are
  not sufficient.
- Image size should be kept as small as possible, without the loss of needed
  details for decision making.
- Images should be made grayscale in situations where color information is not
  needed for making informed decisions.

### Raycast Observations

Raycasts are another possible method for providing observations to an agent.
This can be easily implemented by adding a `RayPerceptionSensorComponent3D` (or
`RayPerceptionSensorComponent2D`) to the Agent GameObject.

During observations, several rays (or spheres, depending on settings) are cast
into the physics world, and the objects that are hit determine the observation
vector that is produced.

![Agent with two RayPerceptionSensorComponent3Ds](images/ray_perception.png)

Both sensor components have several settings:

- _Detectable Tags_ A list of strings corresponding to the types of objects that
  the Agent should be able to distinguish between. For example, in the WallJump
  example, we use "wall", "goal", and "block" as the list of objects to detect.
- _Rays Per Direction_ Determines the number of rays that are cast. One ray is
  always cast forward, and this many rays are cast to the left and right.
- _Max Ray Degrees_ The angle (in degrees) for the outermost rays. 90 degrees
  corresponds to the left and right of the agent.
- _Sphere Cast Radius_ The size of the sphere used for sphere casting. If set to
  0, rays will be used instead of spheres. Rays may be more efficient,
  especially in complex scenes.
- _Ray Length_ The length of the casts
- _Ray Layer Mask_ The [LayerMask](https://docs.unity3d.com/ScriptReference/LayerMask.html)
  passed to the raycast or spherecast. This can be used to ignore certain types
  of objects when casting.
- _Observation Stacks_ The number of previous results to "stack" with the cast
  results. Note that this can be independent of the "Stacked Vectors" setting in
  `Behavior Parameters`.
- _Start Vertical Offset_ (3D only) The vertical offset of the ray start point.
- _End Vertical Offset_ (3D only) The vertical offset of the ray end point.

In the example image above, the Agent has two `RayPerceptionSensorComponent3D`s.
Both use 3 Rays Per Direction and 90 Max Ray Degrees. One of the components had
a vertical offset, so the Agent can tell whether it's clear to jump over the
wall.

The total size of the created observations is

```
(Observation Stacks) * (1 + 2 * Rays Per Direction) * (Num Detectable Tags + 2)
```

so the number of rays and tags should be kept as small as possible to reduce the
amount of data used. Note that this is separate from the State Size defined in
`Behavior Parameters`, so you don't need to worry about the formula above when
setting the State Size.

#### RayCast Observation Summary & Best Practices

- Attach `RayPerceptionSensorComponent3D` or `RayPerceptionSensorComponent2D` to
  use.
- This observation type is best used when there is relevant spatial information
  for the agent that doesn't require a fully rendered image to convey.
- Use as few rays and tags as necessary to solve the problem in order to improve
  learning stability and agent performance.

## Actions

An action is an instruction from the Policy that the agent carries out. The
action is passed to the Agent as the `ActionBuffers` parameter when the Academy invokes the
agent's `OnActionReceived()` function. There are two types of actions that an Agent can use:
 **Continuous** and **Discrete**.

Neither the Policy nor the training algorithm know anything about what the
action values themselves mean. The training algorithm simply tries different
values for the action list and observes the affect on the accumulated rewards
over time and many training episodes. Thus, the only place actions are defined
for an Agent is in the `OnActionReceived()` function.

For example, if you designed an agent to move in two dimensions, you could use
either continuous or the discrete actions. In the continuous case, you
would set the action size to two (one for each dimension), and the
agent's Policy would output an action with two floating point values. In the
discrete case, you would use one Branch with a size of four (one for each
direction), and the Policy would create an action array containing a single
element with a value ranging from zero to three. Alternatively, you could create
two branches of size two (one for horizontal movement and one for vertical
movement), and the Policy would output an action array containing two elements
with values ranging from zero to one. You could alternatively use a combination of continuous
and discrete actions e.g., using one continuous action for horizontal movement
and a discrete branch of size two for the vertical movement.

Note that when you are programming actions for an agent, it is often helpful to
test your action logic using the `Heuristic()` method of the Agent, which lets
you map keyboard commands to actions.

### Continuous Actions

When an Agent's Policy has **Continuous** actions, the
`ActionBuffers.ContinuousActions` passed to the Agent's `OnActionReceived()` function
is an array with length equal to the `Continuous Action Size` property value. The
individual values in the array have whatever meanings that you ascribe to them.
If you assign an element in the array as the speed of an Agent, for example, the
training process learns to control the speed of the Agent through this
parameter.

The [Reacher example](Learning-Environment-Examples.md#reacher) uses
continuous actions with four control values.

![reacher](images/reacher.png)

These control values are applied as torques to the bodies making up the arm:

```csharp
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var torqueX = Mathf.Clamp(actionBuffers.ContinuousActions[0], -1f, 1f) * 150f;
        var torqueZ = Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f) * 150f;
        m_RbA.AddTorque(new Vector3(torqueX, 0f, torqueZ));

        torqueX = Mathf.Clamp(actionBuffers.ContinuousActions[2], -1f, 1f) * 150f;
        torqueZ = Mathf.Clamp(actionBuffers.ContinuousActions[3], -1f, 1f) * 150f;
        m_RbB.AddTorque(new Vector3(torqueX, 0f, torqueZ));
    }
```

By default the output from our provided PPO algorithm pre-clamps the values of
`ActionBuffers.ContinuousActions` into the [-1, 1] range. It is a best practice to manually clip
these as well, if you plan to use a 3rd party algorithm with your environment.
As shown above, you can scale the control values as needed after clamping them.

### Discrete Actions

When an Agent's Policy uses **discrete** actions, the
`ActionBuffers.DiscreteActions` passed to the Agent's `OnActionReceived()` function
is an array of integers with length equal to `Discrete Branch Size`. When defining the discrete actions, `Branches`
is an array of integers, each value corresponds to the number of possibilities for each branch.

For example, if we wanted an Agent that can move in a plane and jump, we could
define two branches (one for motion and one for jumping) because we want our
agent be able to move **and** jump concurrently. We define the first branch to
have 5 possible actions (don't move, go left, go right, go backward, go forward)
and the second one to have 2 possible actions (don't jump, jump). The
`OnActionReceived()` method would look something like:

```csharp
// Get the action index for movement
int movement = actionBuffers.DiscreteActions[0];
// Get the action index for jumping
int jump = actionBuffers.DiscreteActions[1];

// Look up the index in the movement action list:
if (movement == 1) { directionX = -1; }
if (movement == 2) { directionX = 1; }
if (movement == 3) { directionZ = -1; }
if (movement == 4) { directionZ = 1; }
// Look up the index in the jump action list:
if (jump == 1 && IsGrounded()) { directionY = 1; }

// Apply the action results to move the Agent
gameObject.GetComponent<Rigidbody>().AddForce(
    new Vector3(
        directionX * 40f, directionY * 300f, directionZ * 40f));
```

#### Masking Discrete Actions

When using Discrete Actions, it is possible to specify that some actions are
impossible for the next decision. When the Agent is controlled by a neural
network, the Agent will be unable to perform the specified action. Note that
when the Agent is controlled by its Heuristic, the Agent will still be able to
decide to perform the masked action. In order to mask an action, override the
`Agent.WriteDiscreteActionMask()` virtual method, and call
`WriteMask()` on the provided `IDiscreteActionMask`:

```csharp
public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
{
    actionMasker.WriteMask(branch, actionIndices)
}
```

Where:

- `branch` is the index (starting at 0) of the branch on which you want to mask
  the action
- `actionIndices` is a list of `int` corresponding to the indices of the actions
  that the Agent **cannot** perform.

For example, if you have an Agent with 2 branches and on the first branch
(branch 0) there are 4 possible actions : _"do nothing"_, _"jump"_, _"shoot"_
and _"change weapon"_. Then with the code bellow, the Agent will either _"do
nothing"_ or _"change weapon"_ for his next decision (since action index 1 and 2
are masked)

```csharp
WriteMask(0, new int[2]{1,2})
```

Notes:

- You can call `WriteMask` multiple times if you want to put masks on multiple
  branches.
- You cannot mask all the actions of a branch.
- You cannot mask actions in continuous control.

### Actions Summary & Best Practices

- Agents can use `Discrete` and/or `Continuous` actions.
- Discrete actions can have multiple action branches, and it's possible to mask
  certain actions so that they won't be taken.
- In general, fewer actions will make for easier learning.
- Be sure to set the Continuous Action Size and Discrete Branch Size to the desired
  number for each type of action, and not greater, as doing the latter can interfere with the
  efficiency of the training process.
- Continuous action values should be clipped to an
  appropriate range. The provided PPO model automatically clips these values
  between -1 and 1, but third party training systems may not do so.

## Rewards

In reinforcement learning, the reward is a signal that the agent has done
something right. The PPO reinforcement learning algorithm works by optimizing
the choices an agent makes such that the agent earns the highest cumulative
reward over time. The better your reward mechanism, the better your agent will
learn.

**Note:** Rewards are not used during inference by an Agent using a trained
model and is also not used during imitation learning.

Perhaps the best advice is to start simple and only add complexity as needed. In
general, you should reward results rather than actions you think will lead to
the desired results. You can even use the Agent's Heuristic to control the Agent
while watching how it accumulates rewards.

Allocate rewards to an Agent by calling the `AddReward()` or `SetReward()`
methods on the agent. The reward assigned between each decision should be in the
range [-1,1]. Values outside this range can lead to unstable training. The
`reward` value is reset to zero when the agent receives a new decision. If there
are multiple calls to `AddReward()` for a single agent decision, the rewards
will be summed together to evaluate how good the previous decision was. The
`SetReward()` will override all previous rewards given to an agent since the
previous decision.

### Examples

You can examine the `OnActionReceived()` functions defined in the
[example environments](Learning-Environment-Examples.md) to see how those
projects allocate rewards.

The `GridAgent` class in the
[GridWorld example](Learning-Environment-Examples.md#gridworld) uses a very
simple reward system:

```csharp
Collider[] hitObjects = Physics.OverlapBox(trueAgent.transform.position,
                                           new Vector3(0.3f, 0.3f, 0.3f));
if (hitObjects.Where(col => col.gameObject.tag == "goal").ToArray().Length == 1)
{
    AddReward(1.0f);
    EndEpisode();
}
else if (hitObjects.Where(col => col.gameObject.tag == "pit").ToArray().Length == 1)
{
    AddReward(-1f);
    EndEpisode();
}
```

The agent receives a positive reward when it reaches the goal and a negative
reward when it falls into the pit. Otherwise, it gets no rewards. This is an
example of a _sparse_ reward system. The agent must explore a lot to find the
infrequent reward.

In contrast, the `AreaAgent` in the
[Area example](Learning-Environment-Examples.md#push-block) gets a small
negative reward every step. In order to get the maximum reward, the agent must
finish its task of reaching the goal square as quickly as possible:

```csharp
AddReward( -0.005f);
MoveAgent(act);

if (gameObject.transform.position.y < 0.0f ||
    Mathf.Abs(gameObject.transform.position.x - area.transform.position.x) > 8f ||
    Mathf.Abs(gameObject.transform.position.z + 5 - area.transform.position.z) > 8)
{
    AddReward(-1f);
    EndEpisode();
}
```

The agent also gets a larger negative penalty if it falls off the playing
surface.

The `Ball3DAgent` in the
[3DBall](Learning-Environment-Examples.md#3dball-3d-balance-ball) takes a
similar approach, but allocates a small positive reward as long as the agent
balances the ball. The agent can maximize its rewards by keeping the ball on the
platform:

```csharp

SetReward(0.1f);

// When ball falls mark Agent as finished and give a negative penalty
if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
    Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
    Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
{
    SetReward(-1f);
    EndEpisode();

}
```

The `Ball3DAgent` also assigns a negative penalty when the ball falls off the
platform.

Note that all of these environments make use of the `EndEpisode()` method, which
manually terminates an episode when a termination condition is reached. This can
be called independently of the `Max Step` property.

### Rewards Summary & Best Practices

- Use `AddReward()` to accumulate rewards between decisions. Use `SetReward()`
  to overwrite any previous rewards accumulate between decisions.
- The magnitude of any given reward should typically not be greater than 1.0 in
  order to ensure a more stable learning process.
- Positive rewards are often more helpful to shaping the desired behavior of an
  agent than negative rewards. Excessive negative rewards can result in the
  agent failing to learn any meaningful behavior.
- For locomotion tasks, a small positive reward (+0.1) for forward velocity is
  typically used.
- If you want the agent to finish a task quickly, it is often helpful to provide
  a small penalty every step (-0.05) that the agent does not complete the task.
  In this case completion of the task should also coincide with the end of the
  episode by calling `EndEpisode()` on the agent when it has accomplished its
  goal.

## Agent Properties

![Agent Inspector](images/3dball_learning_brain.png)

- `Behavior Parameters` - The parameters dictating what Policy the Agent will
  receive.
  - `Behavior Name` - The identifier for the behavior. Agents with the same
    behavior name will learn the same policy.
  - `Vector Observation`
    - `Space Size` - Length of vector observation for the Agent.
    - `Stacked Vectors` - The number of previous vector observations that will
      be stacked and used collectively for decision making. This results in the
      effective size of the vector observation being passed to the Policy being:
      _Space Size_ x _Stacked Vectors_.
  - `Actions`
    - `Continuous Actions` - The number of concurrent continuous actions that
     the Agent can take.
    - `Discrete Branches` - An array of integers, defines multiple concurrent
      discrete actions. The values in the `Discrete Branches` array correspond
      to the number of possible discrete values for each action branch.
  - `Model` - The neural network model used for inference (obtained after
    training)
  - `Inference Device` - Whether to use CPU or GPU to run the model during
    inference
  - `Behavior Type` - Determines whether the Agent will do training, inference,
    or use its Heuristic() method:
    - `Default` - the Agent will train if they connect to a python trainer,
      otherwise they will perform inference.
    - `Heuristic Only` - the Agent will always use the `Heuristic()` method.
    - `Inference Only` - the Agent will always perform inference.
  - `Team ID` - Used to define the team for self-play
  - `Use Child Sensors` - Whether to use all Sensor components attached to child
    GameObjects of this Agent.
- `Max Step` - The per-agent maximum number of steps. Once this number is
  reached, the Agent will be reset.

## Destroying an Agent

You can destroy an Agent GameObject during the simulation. Make sure that there
is always at least one Agent training at all times by either spawning a new
Agent every time one is destroyed or by re-spawning new Agents when the whole
environment resets.

## Defining Teams for Multi-agent Scenarios

Self-play is triggered by including the self-play hyperparameter hierarchy in
the [trainer configuration](Training-ML-Agents.md#training-configurations). To
distinguish opposing agents, set the team ID to different integer values in the
behavior parameters script on the agent prefab.

<p align="center">
  <img src="images/team_id.png"
       alt="Team ID"
       width="375" border="10" />
</p>

**_Team ID must be 0 or an integer greater than 0._**

In symmetric games, since all agents (even on opposing teams) will share the
same policy, they should have the same 'Behavior Name' in their Behavior
Parameters Script. In asymmetric games, they should have a different Behavior
Name in their Behavior Parameters script. Note, in asymmetric games, the agents
must have both different Behavior Names _and_ different team IDs!

For examples of how to use this feature, you can see the trainer configurations
and agent prefabs for our Tennis and Soccer environments. Tennis and Soccer
provide examples of symmetric games. To train an asymmetric game, specify
trainer configurations for each of your behavior names and include the self-play
hyperparameter hierarchy in both.

## Recording Demonstrations

In order to record demonstrations from an agent, add the
`Demonstration Recorder` component to a GameObject in the scene which contains
an `Agent` component. Once added, it is possible to name the demonstration that
will be recorded from the agent.

<p align="center">
  <img src="images/demo_component.png"
       alt="Demonstration Recorder"
       width="650" border="10" />
</p>

When `Record` is checked, a demonstration will be created whenever the scene is
played from the Editor. Depending on the complexity of the task, anywhere from a
few minutes or a few hours of demonstration data may be necessary to be useful
for imitation learning. To specify an exact number of steps you want to record
use the `Num Steps To Record` field and the editor will end your play session
automatically once that many steps are recorded. If you set `Num Steps To Record`
to `0` then recording will continue until you manually end the play session. Once
the play session ends a `.demo` file will be created in the `Assets/Demonstrations`
folder (by default). This file contains the demonstrations. Clicking on the file will
provide metadata about the demonstration in the inspector.

<p align="center">
  <img src="images/demo_inspector.png"
       alt="Demonstration Inspector"
       width="375" border="10" />
</p>

You can then specify the path to this file in your
[training configurations](Training-Configuration-File.md#behavioral-cloning).
