# Training Using Concurrent Unity Instances

As part of release v0.8, we enabled developers to run concurrent, parallel instances of the Unity executable during training. For certain scenarios, this should speed up the training.  

## How to Run Concurrent Unity Instances During Training

Please refer to the general instructions on (Training ML-Agents)[Training-ML-Agents.md].  In order to run concurrent Unity instances during training, set number of enviornments using the command line option `--num-envs=<n>` when you invoke `mlagents-learn`.

## Considerations

