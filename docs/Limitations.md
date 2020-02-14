# Limitations

## Unity SDK

### Headless Mode

If you enable Headless mode, you will not be able to collect visual observations
from your agents.

### Rendering Speed and Synchronization

Currently the speed of the game physics can only be increased to 100x real-time.
The Academy also moves in time with FixedUpdate() rather than Update(), so game
behavior implemented in Update() may be out of sync with the agent decision
making. See
[Execution Order of Event Functions](https://docs.unity3d.com/Manual/ExecutionOrder.html)
for more information.

You can control the frequency of Academy stepping by calling
`Academy.Instance.DisableAutomaticStepping()`, and then calling
`Academy.Instance.EnvironmentStep()`

### Unity Inference Engine Models
Currently, only models created with our trainers are supported for running
ML-Agents with a neural network behavior.

## Python API

### Python version

As of version 0.3, we no longer support Python 2.

