# About ML-Agents Extensions package (`com.unity.ml-agents.extensions`)

The Unity ML-Agents Extensions package contains optional add-ons to the C# SDK for the
[Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).

These extensions are all considered experimental, and their API or behavior
may change between versions.


## Package contents

The following table describes the package folder structure:

| **Location**     | **Description**                                                        |
| ---------------- | ---------------------------------------------------------------------- |
| _Documentation~_ | Contains the documentation for the Unity package.                      |
| _Editor_         | Contains utilities for Editor windows and drawers.                     |
| _Runtime_        | Contains core C# APIs for integrating ML-Agents into your Unity scene. |
| _Tests_          | Contains the unit tests for the package.                               |

The Runtime directory currently contains these features:
 * Physics-based sensors
 * [Input System Package Integration](InputActuatorComponent.md)
 * [Custom Grid-based Sensors](CustomGridSensors.md)

## Installation
The ML-Agents Extensions package is not currently available in the Package Manager. There are two
recommended ways to install the package:

### Local Installation
[Clone the repository](https://github.com/Unity-Technologies/ml-agents/tree/release_19_docs/docs/Installation.md#clone-the-ml-agents-toolkit-repository-optional) and follow the
[Local Installation for Development](https://github.com/Unity-Technologies/ml-agents/tree/release_19_docs/docs/Installation.md#advanced-local-installation-for-development-1)
directions (substituting `com.unity.ml-agents.extensions` for the package name).

### Github via Package Manager
In Unity 2019.4 or later, open the Package Manager, hit the "+" button, and select "Add package from git URL".

![Package Manager git URL](https://github.com/Unity-Technologies/ml-agents/blob/release_19_docs/docs/images/unity_package_manager_git_url.png)

In the dialog that appears, enter
 ```
git+https://github.com/Unity-Technologies/ml-agents.git?path=com.unity.ml-agents.extensions#release_19
```

You can also edit your project's `manifest.json` directly and add the following line to the `dependencies`
section:
```
"com.unity.ml-agents.extensions": "git+https://github.com/Unity-Technologies/ml-agents.git?path=com.unity.ml-agents.extensions#release_19",
```
See [Git dependencies](https://docs.unity3d.com/Manual/upm-git.html#subfolder) for more information. Note that this
may take several minutes to resolve the packages the first time that you add it.


## Requirements

This version of the Unity ML-Agents package is compatible with the following
versions of the Unity Editor:

- 2019.4 and later

If using the `InputActuatorComponent`
- install the `com.unity.inputsystem` package version `1.1.0-preview.3` or later.

## Known Limitations
- For the `InputActuatorComponent`
    - Limited implementation of `InputControls`
    - No way to customize the action space of the `InputActuatorComponent`

## Need Help?
The main [README](https://github.com/Unity-Technologies/ml-agents/tree/release_19_docs/README.md) contains links for contacting the team or getting support.
