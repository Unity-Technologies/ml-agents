# Unity Environment Registry [Experimental]

The Unity Environment Registry is a database of pre-built Unity environments that can be easily used without having to install the Unity Editor. It is a great way to get started with our [UnityEnvironment API](Python-API.md).

## Loading an Environment from the Registry

To get started, you can access the default registry we provide with our [Example Environments](Learning-Environment-Examples.md). The Unity Environment Registry implements a _Mapping_, therefore, you can access an entry with its identifier with the square brackets `[ ]`. Use the following code to list all of the environment identifiers present in the default registry:

```python
from mlagents_envs.registry import default_registry

environment_names = list(default_registry.keys())
for name in environment_names:
   print(name)
```

The `make()` method on a registry value will return a `UnityEnvironment` ready to be used. All arguments passed to the make method will be passed to the constructor of the `UnityEnvironment` as well. Refer to the documentation on the [Python-API](Python-API.md) for more information about the arguments of the `UnityEnvironment` constructor. For example, the following code will create the environment under the identifier `"my-env"`, reset it, perform a few steps and finally close it:

```python
from mlagents_envs.registry import default_registry

env = default_registry["my-env"].make()
env.reset()
for _ in range(10):
  env.step()
env.close()
```

## Create and share your own registry

In order to share the `UnityEnvironemnt` you created, you must :
 - [Create a Unity executable](Learning-Environment-Executable.md) of your environment for each platform (Linux, OSX and/or Windows)
 - Place each executable in a `zip` compressed folder
 - Upload each zip file online to your preferred hosting platform
 - Create a `yaml` file that will contain the description and path to your environment
 - Upload the `yaml` file online
The `yaml` file must have the following format :

```yaml
environments:
  - <environment-identifier>:
     expected_reward: <expected-reward-float>
     description: <description-of-the-environment>
     linux_url: <url-to-the-linux-zip-folder>
     darwin_url: <url-to-the-osx-zip-folder>
     win_url: <url-to-the-windows-zip-folder>
     additional_args:
      - <an-optional-list-of-command-line-arguments-for-the-executable>
      - ...
```

Your users can now use your environment with the following code :
```python
from mlagents_envs.registry import UnityEnvRegistry

registry = UnityEnvRegistry()
registry.register_from_yaml("url-or-path-to-your-yaml-file")
```
 __Note__: The `"url-or-path-to-your-yaml-file"` can be either a url or a local path.

