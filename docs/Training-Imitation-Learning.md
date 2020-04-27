# Training with Imitation Learning

## Recording Demonstrations

In order to record demonstrations from an agent, add the `Demonstration Recorder`
component to a GameObject in the scene which contains an `Agent` component.
Once added, it is possible to name the demonstration that will be recorded
from the agent.

<p align="center">
  <img src="images/demo_component.png"
       alt="Demonstration Recorder"
       width="375" border="10" />
</p>

When `Record` is checked, a demonstration will be created whenever the scene
is played from the Editor. Depending on the complexity of the task, anywhere
from a few minutes or a few hours of demonstration data may be necessary to
be useful for imitation learning. When you have recorded enough data, end
the Editor play session. A `.demo` file will be created in the
`Assets/Demonstrations` folder (by default). This file contains the demonstrations.
Clicking on the file will provide metadata about the demonstration in the
inspector.

<p align="center">
  <img src="images/demo_inspector.png"
       alt="Demonstration Inspector"
       width="375" border="10" />
</p>

You can then specify the path to this file as the `demo_path` in your `trainer_config.yaml` file
when using BC or GAIL. For instance, for BC:

```
    behavioral_cloning:
        demo_path: <path_to_your_demo_file>
        ...
```
And for GAIL:
```
    reward_signals:
        gail:
            demo_path: <path_to_your_demo_file>
            ...
```
