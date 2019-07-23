# Training Generalized Reinforcement Learning agents

Reinforcement learning has a rather unique setup as opposed to supervised and
unsupervised learning. Agents here are trained and tested on the same exact 
environment, which is analogous to a model being trained and tested on an 
identical dataset in supervised learning! This setting results in overfitting; 
the inability of the agent to generalize to slight tweaks or variations in the 
environment. This is problematic in instances when environments are randomly 
instantiated with varying properties. To make agents more robust, we train an 
agent over multiple variations of the environment. The agent is trained with 
the intent that it learns to maintain a minimum performance regardless of the 
environment variant and that it generalizes to maintain this in unseen future 
variants of the environment.

Ball scale of 0.5          |  Ball scale of 4
:-------------------------:|:-------------------------:
![](images/3dball_small.png)  |  ![](images/3dball_big.png)

_Variations of the 3D Ball environment._

To vary environments, we first decide what parameters to vary in an
environment. These parameters are known as `Reset Parameters`. In the 3D ball 
environment example displayed in the figure above, the reset parameters are `gravity`, `ball_mass` and `ball_scale`.


## How-to

For generalization training, we need to provide a way to modify the environment 
by supplying a set of reset parameters. This provision can be either 
deterministic or randomized. Each reset parameter is assigned a sampler. If a 
sampler isn't provided for a reset parameter, the parameter maintains the 
default value throughout the training, remaining unchanged. The samplers for all 
the reset parameters are handled by a **Sampler Manager**, which is also 
responsible for generating a new set of values for the reset parameters when 
needed. 

To setup the Sampler Manager, we setup a YAML file that specifies how we wish to 
generate new samples. In this file, we specify the samplers and the 
`resampling-duration` (number of training steps after which reset parameters are 
resampled). Below is an example of a sampler file for the 3D ball environment.

```yaml
episode-length: 5000

mass:
    sampler-type: "uniform"
    min_value: 0.5
    max_value: 10

gravity:
    sampler-type: "multirange_uniform"
    intervals: [[7, 10], [15, 20]]

scale:
    sampler-type: "uniform"
    min_value: 0.75
    max_value: 3

```

* `resampling-duration` (int) - Specifies the number of steps for agent to 
train under a particular environment configuration before resetting the 
environment with a new sample of reset parameters.

* `parameter_name` - Name of the reset parameter. This should match the name 
specified in the academy of the intended environment for which the agent is 
being trained. If a parameter specified in the file doesn't exist in the 
environment, then this specification will be ignored.

    * `sampler-type` - Specify the sampler type to use for the reset parameter. 
    This is a string that should exist in the `Sampler Factory` (explained 
    below).

    * `sub-arguments` - Specify the characteristic parameters for the sampler. 
    In the example sampler file above, this would correspond to the `intervals` 
    key under the `multirange_uniform` sampler for the gravity reset parameter. 
    The key name should match the name of the corresponding argument in the sampler definition. (Look at defining a new sampler method)

The sampler manager allocates a sampler for a reset parameter by using the *Sampler Factory*, which maintains a dictionary mapping of string keys to sampler objects. The available samplers to be used for reset parameter resampling is as available in the Sampler Factory.

The implementation of the samplers can be found at `ml-agents-envs/mlagents/envs/sampler_class.py`.

### Defining a new sampler method

Custom sampling techniques must inherit from the *Sampler* base class (included in the `sampler_class` file) and preserve the interface. Once the class for the required method is specified, it must be registered in the Sampler Factory. 

This can be done by subscribing to the *register_sampler* method of the SamplerFactory. The command is as follows:

`SamplerFactory.register_sampler(*custom_sampler_string_key*, *custom_sampler_object*)`

Once the Sampler Factory reflects the new register, the custom sampler can be used for resampling reset parameter. For demonstration, lets say our sampler was implemented as below, and we register the `CustomSampler` class with the string `custom-sampler` in the Sampler Factory.

```python
class CustomSampler(Sampler):

    def __init__(self, argA, argB, argC):
        self.possible_vals = [argA, argB, argC]

    def sample_all(self):
        return np.random.choice(self.possible_vals)
```

Now we need to specify this sampler in the sampler file. Lets say we wish to use this sampler for the reset parameter *mass*; the sampler file would specify the same for mass as the following.

```yaml
mass:
    sampler-type: "custom-sampler"
    argA: 1
    argB: 2
    argC: 3
```

With the sampler file setup, we can proceed to train our agent as explained in the next section.

### Training with Generalization Learning

We first begin with setting up the sampler file. After the sampler file is defined and configured, we proceed by launching `mlagents-learn` and specify our configured sampler file with the `--sampler` flag. To demonstrate, if we wanted to train a 3D ball agent with generalization using the `generalization-test.yaml` sampling setup, we can run

```sh
mlagents-learn config/trainer_config.yaml --sampler=config/generalize_test.yaml --run-id=3D-Ball-generalization --train
```

We can observe progress and metrics via Tensorboard.
