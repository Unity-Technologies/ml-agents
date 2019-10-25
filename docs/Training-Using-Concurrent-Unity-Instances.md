# Training Using Concurrent Unity Instances

As part of release v0.8, we enabled developers to run concurrent, parallel instances of the Unity executable during training. For certain scenarios, this should speed up the training.

## How to Run Concurrent Unity Instances During Training

Please refer to the general instructions on [Training ML-Agents](Training-ML-Agents.md).  In order to run concurrent Unity instances during training, set the number of environment instances using the command line option `--num-envs=<n>` when you invoke `mlagents-learn`. Optionally, you can also set the `--base-port`, which is the starting port used for the concurrent Unity instances.

## Considerations

### Buffer Size

If you are having trouble getting an agent to train, even with multiple concurrent Unity instances, you could increase  `buffer_size` in the `config/trainer_config.yaml` file. A common practice is to multiply `buffer_size` by `num-envs`.

### Resource Constraints

Invoking concurrent Unity instances is constrained by the resources on the machine.  Please use discretion when setting `--num-envs=<n>`.

### Using num-runs and num-envs

If you set `--num-runs=<n>` greater than 1 and are also invoking concurrent Unity instances using `--num-envs=<n>`, then the number of concurrent Unity instances is equal to `num-runs` times `num-envs`.

### Result Variation Using Concurrent Unity Instances

If you keep all the hyperparameters the same, but change `--num-envs=<n>`, the results and model would likely change.
