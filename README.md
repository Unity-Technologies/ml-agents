<img src="docs/images/unity-wide.png" align="middle" width="3000"/>

# Unity ML-Agents (Beta)

**Unity Machine Learning Agents** (ML-Agents) is an open-source Unity plugin 
that enables games and simulations to serve as environments for training
intelligent agents. Agents can be trained using reinforcement learning,
imitation learning, neuroevolution, or other machine learning methods through
a simple-to-use Python API. We also provide implementations (based on
TensorFlow) of state-of-the-art algorithms to enable game developers
and hobbyists to easily train intelligent agents for 2D, 3D and VR/AR games.
These trained agents can be used for multiple purposes, including
controlling NPC behavior (in a variety of settings such as multi-agent and
adversarial), automated testing of game builds and evaluating different game
design decisions pre-release. ML-Agents is mutually beneficial for both game
developers and AI researchers as it provides a central platform where advances
in AI can be evaluated on Unity’s rich environments and then made accessible
to the wider research and game developer communities. 

## Features
* Unity environment control from Python
* 10+ sample Unity environments
* Support for multiple environment configurations and training scenarios
* Train memory-enhanced Agents using deep reinforcement learning
* Easily definable Curriculum Learning scenarios
* Broadcasting of Agent behavior for supervised learning
* Built-in support for Imitation Learning
* Flexible Agent control with On Demand Decision Making
* Visualizing network outputs within the environment
* Simplified set-up with Docker (Experimental)

## Quick Start Guide

### Get **Unity**

[Download](https://store.unity.com/download) and install Unity, ML-Agents works with versions that are 2017.1 and above. 

### Get Python 3 environment

[Download](https://www.python.org/downloads/release/python-354/) and install Python 3.5. 

### Clone the repo and install dependencies

You can get the ML-Agents code and all the packages it depends on by running the following in your command line tool. 

```
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents/python
pip install .
```

(For Windows) If you don't have git, download it [here](https://git-scm.com/download/win) and install it, then reopen your cmd and try again. 

### Setting up ML-Agent within Unity

1. Open Unity and use it to open folder `ml-agents`/`unity-environment`. 
2. Go to `Edit` -> `Project Settings` -> `Player`
3. For each of the platforms you target 
(**`PC, Mac and Linux Standalone`**, **`iOS`** or **`Android`**):
    1. Go into `Other Settings`.
    2. Select `Scripting Runtime Version` to 
    `Experimental (.NET 4.6 Equivalent)`
    3. In `Scripting Defined Symbols`, add the flag `ENABLE_TENSORFLOW`. 
    After typing in, press Enter.
4. Go to `File` -> `Save Project`
5. Restart the Unity Editor.

### Setting up TensorflowSharp Support

[Download](https://s3.amazonaws.com/unity-ml-agents/0.3/TFSharpPlugin.unitypackage) TensorflowSharp plugin, and import it into Unity once downloaded by double clicking on it.  You can see if it was successfully imported by checking the TensorFlow files in the Project window under `Assets` -> `ML-Agents` -> `Plugins` -> `Computer`. 

**Note**: If you don't see anything under `Assets`, drag the `ml-agents`/`unity-environment`/`Assets`/`ML-Agents` folder under `Assets` within Project window.

### Play an example environment using pretrained model

1. In the Project window, go to `Assets` -> `ML-Agents` -> `Examples` -> `3DBall` folder and open the `3DBall` scene file. 
2. In the Hierarchy window, click on `Ball3DAcademy` -> `Ball3DBrain`. 
3. In the Inspector window, under `Brain (Script)` -> `Brain Type`, change the `Brain Type` to `Internal`. 
4. Click the `Play` button and you will see the platforms automatically adjusts itself using the pretrained model.

## Documentation and References

You can find the complete documentation [here](docs/Readme.md). 

If you have
used a version of ML-Agents prior to v0.3, we strongly recommend 
our [guide on migrating to v0.3](docs/Migrating-v0.3.md).

We have also published a series of blog posts that are relevant for ML-Agents:
- Overviewing reinforcement learning concepts
([multi-armed bandit](https://blogs.unity3d.com/2017/06/26/unity-ai-themed-blog-entries/)
and [Q-learning](https://blogs.unity3d.com/2017/08/22/unity-ai-reinforcement-learning-with-q-learning/))
- [Using Machine Learning Agents in a real game: a beginner’s guide](https://blogs.unity3d.com/2017/12/11/using-machine-learning-agents-in-a-real-game-a-beginners-guide/)
- [Post](https://blogs.unity3d.com/2018/02/28/introducing-the-winners-of-the-first-ml-agents-challenge/) announcing the winners of our
[first ML-Agents Challenge](https://connect.unity.com/challenges/ml-agents-1)
- [Post](https://blogs.unity3d.com/2018/01/23/designing-safer-cities-through-simulations/)
overviewing how Unity can be leveraged as a simulator to design safer cities.

In addition to our own documentation, here are some additional, relevant articles:
- [Unity AI - Unity 3D Artificial Intelligence](https://www.youtube.com/watch?v=bqsfkGbBU6k)
- [A Game Developer Learns Machine Learning](https://mikecann.co.uk/machine-learning/a-game-developer-learns-machine-learning-intent/)
- [Explore Unity Technologies ML-Agents Exclusively on Intel Architecture](https://software.intel.com/en-us/articles/explore-unity-technologies-ml-agents-exclusively-on-intel-architecture)

## Community and Feedback

ML-Agents is an open-source project and we encourage and welcome contributions.
If you wish to contribute, be sure to review our 
[contribution guidelines](CONTRIBUTING.md) and 
[code of conduct](CODE_OF_CONDUCT.md).

You can connect with us and the broader community
through Unity Connect and GitHub:
* Join our
[Unity Machine Learning Channel](https://connect.unity.com/messages/c/035fba4f88400000)
to connect with others using ML-Agents and Unity developers enthusiastic
about machine learning. We use that channel to surface updates
regarding ML-Agents (and, more broadly, machine learning in games).
* If you run into any problems using ML-Agents, 
[submit an issue](https://github.com/Unity-Technologies/ml-agents/issues) and
make sure to include as much detail as possible.

For any other questions or feedback, connect directly with the ML-Agents
team at ml-agents@unity3d.com.

## License

[Apache License 2.0](LICENSE)
