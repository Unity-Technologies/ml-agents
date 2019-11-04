# Training with Behavioral Cloning

There are a variety of possible imitation learning algorithms which can
be used, the simplest one of them is Behavioral Cloning. It works by collecting
demonstrations from a teacher, and then simply uses them to directly learn a
policy, in the same way the supervised learning for image classification
or other traditional Machine Learning tasks work.

## Offline Training

With offline behavioral cloning, we can use demonstrations (`.demo` files)
generated using the `Demonstration Recorder` as the dataset used to train a behavior.

1. Choose an agent you would like to learn to imitate some set of demonstrations.
2. Record a set of demonstration using the `Demonstration Recorder` (see [here](Training-Imitation-Learning.md)).
   For illustrative purposes we will refer to this file as `AgentRecording.demo`.
3. Build the scene(make sure the Agent is not using its heuristic).
4. Open the `config/offline_bc_config.yaml` file.
5. Modify the `demo_path` parameter in the file to reference the path to the
   demonstration file recorded in step 2. In our case this is:
   `./UnitySDK/Assets/Demonstrations/AgentRecording.demo`
6. Launch `mlagent-learn`, providing `./config/offline_bc_config.yaml`
   as the config parameter, and include the `--run-id` and `--train` as usual.
   Provide your environment as the `--env` parameter if it has been compiled
   as standalone, or omit to train in the editor.
7. (Optional) Observe training performance using TensorBoard.

This will use the demonstration file to train a neural network driven agent
to directly imitate the actions provided in the demonstration. The environment
will launch and be used for evaluating the agent's performance during training.
