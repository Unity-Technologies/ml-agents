## Package Limitations
### Training
Training is limited to the Unity Editor and Standalone builds on Windows, MacOS, and Linux with the Mono scripting backend. Currently, training does not work with the IL2CPP scripting backend. Your environment will default to inference mode if training is not supported or is not currently running.

### Inference
Inference is executed via Unity Inference Engine on the end-user device. Therefore, it is subject to the performance limitations of the end-user CPU or GPU. Also, only models created with our trainers are supported for running ML-Agents with a neural network behavior.

### Headless Mode
If you enable Headless mode, you will not be able to collect visual observations from your agents.

### Rendering Speed and Synchronization
Currently the speed of the game physics can only be increased to 100x real-time. The Academy (the sentinel that controls the stepping of the game to make sure everything is synchronized, from collection of observations to applying actions generated from policy inference to the agent) also moves in time with `FixedUpdate()` rather than `Update()`, so game behavior implemented in `Update()` may be out of sync with the agent decision-making. See [Execution Order of Event Functions](https://docs.unity3d.com/Manual/execution-order.html) for more information.

You can control the frequency of Academy stepping by calling `Academy.Instance.DisableAutomaticStepping()`, and then calling `Academy.Instance.EnvironmentStep()`.

### Input System Integration

For `InputActuatorComponent` (see [Input System Integration](InputSystem-Integration.md) for more information):
- Limited implementation of `InputControls`
- No way to customize the action space of the `InputActuatorComponent`
