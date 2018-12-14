# Background: TensorFlow

As discussed in our
[machine learning background page](Background-Machine-Learning.md),
many of the algorithms we provide in the
ML-Agents toolkit leverage some form of deep learning. More specifically, our
implementations are built on top of the open-source library
[TensorFlow](https://www.tensorflow.org/). This means that the models produced
by the ML-Agents toolkit are (currently) in a format only understood by
TensorFlow. In this page we provide a brief overview of TensorFlow, in addition
to TensorFlow-related tools that we leverage within the ML-Agents toolkit.

## TensorFlow

[TensorFlow](https://www.tensorflow.org/) is an open source library for
performing computations using data flow graphs, the underlying representation of
deep learning models. It facilitates training and inference on CPUs and GPUs in
a desktop, server, or mobile device. Within the ML-Agents toolkit, when you
train the behavior of an agent, the output is a TensorFlow model (.bytes) file
that you can then embed within a Learning Brain. Unless you implement a new
algorithm, the use of TensorFlow is mostly abstracted away and behind the
scenes.

## TensorBoard

One component of training models with TensorFlow is setting the values of
certain model attributes (called _hyperparameters_). Finding the right values of
these hyperparameters can require a few iterations. Consequently, we leverage a
visualization tool within TensorFlow called
[TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard).
It allows the visualization of certain agent attributes (e.g. reward) throughout
training which can be helpful in both building intuitions for the different
hyperparameters and setting the optimal values for your Unity environment. We
provide more details on setting the hyperparameters in later parts of the
documentation, but, in the meantime, if you are unfamiliar with TensorBoard we
recommend our guide on [using Tensorboard with ML-Agents](Using-Tensorboard.md) or
this [tutorial](https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial).

## TensorflowSharp

One of the drawbacks of TensorFlow is that it does not provide a native C# API.
This means that the Learning Brain is not natively supported since Unity scripts
are written in C#. Consequently, to enable the Learning Brain, we leverage a
third-party library
[TensorFlowSharp](https://github.com/migueldeicaza/TensorFlowSharp) which
provides .NET bindings to TensorFlow. Thus, when a Unity environment that
contains a Learning Brain is built, inference is performed via TensorFlowSharp.
We provide an additional in-depth overview of how to leverage
[TensorFlowSharp within Unity](Using-TensorFlow-Sharp-in-Unity.md)
which will become more
relevant once you install and start training behaviors within the ML-Agents
toolkit. Given the reliance on TensorFlowSharp, the Learning Brain is currently
marked as experimental.
