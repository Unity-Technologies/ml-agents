# ML-Agents Input System Package Example

### Overview
This small example project shows how to integrate games that use the input system package with ML-Agents. This modified PushBlock scene has a PlayerController script which handles all of the input to control the cube.  On it is an [`InputActuatorComponent`](../com.unity.ml-agents.extensions/Documentation~/InputActuatorComponent.md) which takes the actions associated with the gameObject from the `PlayInput` component and enables the simulation of input data in order to train and run inference with ml-agents.


### See Also
- [ML-Agents Extensions Package](../com.unity.ml-agents.extensions/README.md)
- [Input System Package](https://docs.unity3d.com/Packages/com.unity.inputsystem@1.1/manual/QuickStartGuide.html)
