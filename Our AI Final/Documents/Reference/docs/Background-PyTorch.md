# Background: PyTorch

As discussed in our
[machine learning background page](Background-Machine-Learning.md), many of the
algorithms we provide in the ML-Agents Toolkit leverage some form of deep
learning. More specifically, our implementations are built on top of the
open-source library [PyTorch](https://pytorch.org/). In this page we
provide a brief overview of PyTorch and TensorBoard
that we leverage within the ML-Agents Toolkit.

## PyTorch

[PyTorch](https://pytorch.org/) is an open source library for
performing computations using data flow graphs, the underlying representation of
deep learning models. It facilitates training and inference on CPUs and GPUs in
a desktop, server, or mobile device. Within the ML-Agents Toolkit, when you
train the behavior of an agent, the output is a model (.onnx) file that you can
then associate with an Agent. Unless you implement a new algorithm, the use of
PyTorch is mostly abstracted away and behind the scenes.

## TensorBoard

One component of training models with PyTorch is setting the values of
certain model attributes (called _hyperparameters_). Finding the right values of
these hyperparameters can require a few iterations. Consequently, we leverage a
visualization tool called
[TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard).
It allows the visualization of certain agent attributes (e.g. reward) throughout
training which can be helpful in both building intuitions for the different
hyperparameters and setting the optimal values for your Unity environment. We
provide more details on setting the hyperparameters in the
[Training ML-Agents](Training-ML-Agents.md) page. If you are unfamiliar with
TensorBoard we recommend our guide on
[using TensorBoard with ML-Agents](Using-Tensorboard.md) or this
[tutorial](https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial).
