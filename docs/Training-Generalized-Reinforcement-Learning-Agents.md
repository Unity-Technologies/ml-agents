# Training Generalized Reinforcement Learning Agents

One of the challenges of training and testing agents on the same
environment is that the agents tend to overfit. The result is that the
agents are unable to generalize to any tweaks or variations in the enviornment.
This is analgous to a model being trained and tested on an identical dataset
in supervised learning. This becomes problematic in cases where environments
are randomly instantiated with varying objects or properties. 

To make agents robust and generalizable to different environments, the agent
should be trained over multiple variations of the enviornment. Using this approach
for training, the agent will be better suited to adapt (with higher performance)
to future unseen variations of the enviornment

_Variations of the 3D Ball environment._

Ball scale of 0.5          |  Ball scale of 4
:-------------------------:|:-------------------------:
![](images/3dball_small.png)  |  ![](images/3dball_big.png)

## Introducing Generalization Using Reset Parameters

To enable variations in the environments, we implemented `Reset Parameters`. We
also specify a range of values for each `Reset Parameter` and sample
these parameters during training. In the 3D ball environment example displayed
in the figure above, the reset parameters are `gravity`, `ball_mass` and `ball_scale`.


## How to Enable Generalization Using Reset Parameters

We need to provide a way to modify the environment by supplying a set of `Reset Parameters`,
and vary them over time. This provision can be done either deterministically or randomly. 

This is done by assigning each reset parameter a sampler, which samples a reset
parameter value (such as a uniform sampler). If a sampler isn't provided for a
`Reset Parameter`, the parameter maintains the default value throughout the 
training procedure, remaining unchanged. The samplers for all the `Reset Parameters` 
are handled by a **Sampler Manager**, which also handles the generation of new 
values for the reset parameters when needed. 

To setup the Sampler Manager, we create a YAML file that specifies how we wish to 
generate new samples for each `Reset Parameters`. In this file, we specify the samplers and the 
`resampling-interval` (the number of simulation steps after which reset parameters are 
resampled). Below is an example of a sampler file for the 3D ball environment.

```yaml
resampling-interval: 5000

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

Below is the explanation of the fields in the above example.

* `resampling-interval` - Specifies the number of steps for the agent to 
train under a particular environment configuration before resetting the 
environment with a new sample of `Reset Parameters`.

* `parameter_name` - Name of the `Reset Parameter`. This should match the name 
specified in the academy of the intended environment for which the agent is 
being trained. If a parameter specified in the file doesn't exist in the 
environment, then this parameter will be ignored.

    * `sampler-type` - Specify the sampler type to use for the `Reset Parameter`. 
    This is a string that should exist in the `Sampler Factory` (explained 
    below).

    * `sub-arguments` - Specify the sub-arguments depending on the `sampler-type`. 
    In the example above, this would correspond to the `intervals` 
    under the `sampler-type` `multirange_uniform` for the `Reset Parameter` called gravity`. 
    The key name should match the name of the corresponding argument in the sampler definition. 
    (See below)

The Sampler Manager allocates a sampler for each `Reset Parameter` by using the *Sampler Factory*,
which maintains a dictionary mapping of string keys to sampler objects. The available samplers
to be used for each `Reset Parameter` is available in the Sampler Factory.

#### Possible Sampler Types

The currently implemented samplers that can be used with the `sampler-type` arguments are:

* `uniform` - Uniform sampler
    *   Uniformly samples a single float value between defined endpoints. 
        The sub-arguments for this sampler to specify the interval 
        endpoints are as below. The sampling is done in the range of 
        [`min_value`, `max_value`).

    * **sub-arguments** - `min_value`, `max_value`

* `gaussian` - Gaussian sampler 
    *   Samples a single float value from the distribution characterized by
        the mean and standard deviation. The sub-arguments to specify the 
        gaussian distribution to use are as below.

    * **sub-arguments** - `mean`, `st_dev`

* `multirange_uniform` - Multirange Uniform sampler
    *   Uniformly samples a single float value between the specified intervals. 
        Samples by first performing a weight pick of an interval from the list 
        of intervals (weighted based on interval width) and samples uniformly 
        from the selected interval (half-closed interval, same as the uniform 
        sampler). This sampler can take an arbitrary number of intervals in a 
        list in the following format: 
    [[`interval_1_min`, `interval_1_max`], [`interval_2_min`, `interval_2_max`], ...]
    
    * **sub-arguments** - `intervals`


The implementation of the samplers can be found at `ml-agents-envs/mlagents/envs/sampler_class.py`.

### Defining a new sampler method

Custom sampling techniques must inherit from the *Sampler* base class (included in the `sampler_class` file) and preserve the interface. Once the class for the required method is specified, it must be registered in the Sampler Factory. 

This can be done by subscribing to the *register_sampler* method of the SamplerFactory. The command is as follows:

`SamplerFactory.register_sampler(*custom_sampler_string_key*, *custom_sampler_object*)`

Once the Sampler Factory reflects the new register, the custom sampler can be used for resampling `Reset Parameter`. For demonstration, lets say our sampler was implemented as below, and we register the `CustomSampler` class with the string `custom-sampler` in the Sampler Factory.

```python
class CustomSampler(Sampler):

    def __init__(self, argA, argB, argC):
        self.possible_vals = [argA, argB, argC]

    def sample_all(self):
        return np.random.choice(self.possible_vals)
```

Now we need to specify this sampler in the sampler file. Lets say we wish to use this sampler for the `Reset Parameter` *mass*; the sampler file would specify the same for mass as the following (any order of the subarguments is valid).

```yaml
mass:
    sampler-type: "custom-sampler"
    argB: 1
    argA: 2
    argC: 3
```

With the sampler file setup, we can proceed to train our agent as explained in the next section.

### Training with Generalization Learning

We first begin with setting up the sampler file. After the sampler file is defined and configured, we proceed by launching `mlagents-learn` and specify our configured sampler file with the `--sampler` flag. To demonstrate, if we wanted to train a 3D ball agent with generalization using the `config/3dball_generalize.yaml` sampling setup, we can run

```sh
mlagents-learn config/trainer_config.yaml --sampler=config/3dball_generalize.yaml --run-id=3D-Ball-generalization --train
```

We can observe progress and metrics via Tensorboard.
