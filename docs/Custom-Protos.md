# Creating custom protobuffer messages

Unity and Python communicate by sending protobuffer messages to and from each other. You can create custom protobuffer messages if you want to exchange structured data beyond is included by default. 

Whenever you change the fields of a custom message, you must run `protobuf-definitions/make.bat` to create C# and Python files corresponding to the new message. Follow the directions in that file for guidance. After running it, reinstall the Python package by running `pip install ml-agents` and make sure your Unity project is using the newly-generated version of `UnitySDK`.

## Custom message types

There are three custom message types currently supported, described below. In each case, `env` is an instance of a `UnityEnvironment` in Python. `CustomAction` is described most thoroughly; usage of the  other custom messages follows a similar template. 

### Custom actions

By default, the Python API sends actions to Unity in the form of a floating-point list per agent and an optional string-valued text action. 

You can define a custom action type to replace or augment this by adding fields to the `CustomAction` message, which you can do by editing the file "protobuf-definitions/proto/mlagents/envs/communicator_objects/custom_action.proto". 

Instances of custom actions are set via the `custom_action` parameter of `environment.step`. An agent receives a custom action by defining a method with the signature

```csharp
public virtual void AgentAction(float[] vectorAction, string textAction, CommunicatorObjects.CustomAction customAction)
```

Here is an example of creating a custom action that instructs an agent to choose a cardinal direction to walk in and how far to walk. 

`custom_action.proto` will look like 

```protobuf
syntax = "proto3";

option csharp_namespace = "MLAgents.CommunicatorObjects";
package communicator_objects;

message CustomAction {
    enum Direction {
        NORTH=0;
        SOUTH=1;
        EAST=1;
        WEST=1;
    }
    float walkAmount = 1;    
    Direction direction = 2;
}
```

In your Python file, create an instance of a custom action:

```python
env = mlagents.UnityEnvironment(...)
...
action = mlagents.CustomAction(direction=mlagents.CustomAction.NORTH, walkAmount=2.0)
env.step(custom_action=action)
```

Then in your agent,

```csharp
class MyAgent : Agent {
    ...
    virtual void AgentAction(float[] vectorAction, string textAction, CommunicatorObjects.CustomAction action) {
        if(action.direction == CommnicatorObjects.CustomAction.NORTH) {
            translate(0, 0, action.walkAmount); 
        }
        ...
    }
}
```

### Custom reset parameters

By default, you can configure an environment `env ` in the Python API by specifying a `config` parameter that is a dictionary mapping strings to floats. 

You can also configure an environment using a custom protobuf message. To do so, add fields to the `CustomResetParameters` protobuffer message in `custom_reset_parameters.proto`, analogously to `CustomAction` above. Then pass an instance of the message to `env.reset` via the `custom_reset_parameters` keyword parameter.

### Custom observations

By default, Unity returns observations to Python in the form of a floating-point vector. 

You can define a custom observation message to supplement that. To do so, add fields to the `CustomObservation` protobuffer message in `custom_observation.proto`. 

Then in your agent, create an instance of a custom observation via `new CommunicatorObjects.CustomObservation`. Then in `CollectObservations`, call `SetCustomObservation` wit the custom observation instance as the parameter.

In Python, the custom observation can be accessed in the return of `env.step` as ` custom_observations` property (one entry per agent).
