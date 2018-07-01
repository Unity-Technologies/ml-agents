# MujocoUnity

![MujocoUnity](images/MujocoUnityBanner.gif)

*Mujoco* is a high end physics simulator used for bleeding edge research into robotics and reinforcement learning. Many of the standard benchmarks are implemented in Mujoco. 

*MujocoUnity* enables the reproduction of these benchmarks within Unity ml-agents using Unity’s native physics simulator, PhysX. Mujoco Unity maybe useful for:
* Video Game researchers interested in apply bleeding edge robotics research into the domain of locomotion and AI for video games.
* Traditional academic researchers looking to leverage the strengths of Unity and ml-agents along with the body of existing research and benchmarks in Mujoco.
* Benchmarking current and future algorithms within Unity ml-agents. For example, comparing the performance of ml-agents PPO implementation with OpenAI.Baselines implementation of PPO.

### References:
* OpenAI Baselines / Gym / Roboschool
* DeepMind 
* Some other algos

### Important: 
* PhysX makes many tradeoffs in terms of accuracy when compared with Mujoco. It may not be the best choice for your research project.
* MujocoUnity environments are running at 300-500 physics simulations per second. This is significantly higher that Unity’s defaults setting of 50 physics simulations per second.
* Currently, MujocoUnity does not properly simulate how Mujoco handles joint observations - as such, it maybe difficult to do transfer learning (from simulation to real world robots)
* A good primer on the differences between physics engines is ['Physics simulation engines have traditional made tradeoffs between performance’](https://homes.cs.washington.edu/~todorov/papers/ErezICRA15.pdf) and it’s accompanying [video](https://homes.cs.washington.edu/~todorov/media/ErezICRA15.mp4)


## Humanoid


| **DeepMindHumanoid** | **OpenAIHumanoid** |
| --- | --- |
| ![DeepMindHumanoid](images/DeepMindHumanoid102-2m.gif) | ![OpenAIHumanoid](images/OpenAIHumanoid164-2m.gif) |


* Set-up: Simple (OpenAI) and complex (DeepMind) Humanoid agents. 
* Goal: The agent must move its body toward the goal as quickly as possible without falling.
* Agents: The environment contains 16 independent agents linked to a single brain.
* Agent Reward Function: 
  * Reference OpenAI.Roboschool and / or DeepMind
    * -joints at limit penality
    * -effort penality (ignors hip_y and knee)
    * +velocity
    * -height penality if below 1.2m
  * Inspired by Deliberate Practice (currently, only does legs)
    * +facing upright bonus for shoulders, waist, pelvis
    * +facing target bonus for shoulders, waist, pelvis
    * -non straight thigh penality
    * +leg phase bonus (for height of knees)
    * +0.01 times body direction alignment with goal direction.
    * -0.01 times head velocity difference from body velocity.
* Agent Terminate Function: 
  * TerminateOnNonFootHitTerrain - Agent terminates when a body part other than foot collides with the terrain.
* Brains: One brain with the following observation/action space.
    * Vector Observation space: (Continuous) 74 (Simple Humanoid), 88 (Complex Humanoid) variables
    * Vector Action space: (Continuous) Size of 17 (Simple Humanoid), 21 (Complex Humanoid) corresponding to target rotations applicable to the joints. 
    * Visual Observations: None.
* Reset Parameters: None.
* Benchmark Mean Reward: **TODO show vs OpenAI PPO**


## Hopper

| **DeepMindHopper** | **OpenAIHopper** |
| --- | --- |
| ![DeepMindHopper](images/DeepMindHopper101-1m.gif) | ![OpenAIHopper](images/OpenAIHopper102-300k.gif) |

* Set-up: OpenAI and DeepMind Hopper agents. 
* Goal: The agent must move its body toward the goal as quickly as possible without falling.
* Agents: The environment contains 16 independent agents linked to a single brain.
* Agent Reward Function: 
  * Reference OpenAI.Roboschool and / or DeepMind
    * -effort penality
    * +velocity
    * +uprightBonus
    * -height penality if below .65m OpenAI, 1.1m DeepMind
* Agent Terminate Function: 
  * DeepMindHopper: TerminateOnNonFootHitTerrain - Agent terminates when a body part other than foot collides with the terrain.
  * OpenAIHopper
    * TerminateOnNonFootHitTerrain
    * Terminate if height < .3m
    * Terminate if head tilt > 0.4
* Brains: One brain with the following observation/action space.
    * Vector Observation space: (Continuous) 26 (OpenAIHopper), 31 (DeepMindHopper) variables
    * Vector Action space: (Continuous) Size of 3 (OpenAIHopper), 4 (DeepMindHopper) corresponding to target rotations applicable to the joints. 
    * Visual Observations: None.
* Reset Parameters: None.


## Walker

| **DeepMindWalker** | **OpenAIWalker** |
| --- | --- |
| ![DeepMindWalker](images/DeepMindWalker108-1m.gif) | ![OpenAIWalker](images/OpenAIWalker106-300k.gif) |

* Set-up: OpenAI and DeepMind Walker agents. 
* Goal: The agent must move its body toward the goal as quickly as possible without falling.
* Agents: The environment contains 16 independent agents linked to a single brain.
* Agent Reward Function: 
  * Reference OpenAI.Roboschool and / or DeepMind
    * -effort penality
    * +velocity
    * +uprightBonus
    * -height penality if below .65m OpenAI, 1.1m DeepMind
* Agent Terminate Function: 
  * TerminateOnNonFootHitTerrain - Agent terminates when a body part other than foot collides with the terrain.
* Brains: One brain with the following observation/action space.
    * Vector Observation space: (Continuous) 41 variables
    * Vector Action space: (Continuous) Size of 6 corresponding to target rotations applicable to the joints. 
    * Visual Observations: None.
* Reset Parameters: None.

## Ant

| **OpenAIAnt** |
| --- | 
| ![OpenAIAnt](images/OpenAIAnt102-1m.gif) | 

* Set-up: OpenAI and Ant agent. 
* Goal: The agent must move its body toward the goal as quickly as possible without falling.
* Agents: The environment contains 16 independent agents linked to a single brain.
* Agent Reward Function: 
  * Reference OpenAI.Roboschool and / or DeepMind
    * -joints at limit penality
    * -effort penality 
    * +velocity
* Agent Terminate Function: 
  * Terminate if head body > 0.2
* Brains: One brain with the following observation/action space.
    * Vector Observation space: (Continuous) 53 variables
    * Vector Action space: (Continuous) Size of 8 corresponding to target rotations applicable to the joints. 
    * Visual Observations: None.
* Reset Parameters: None.



## Details
### Key Files / Folders
* MujocoUnity - parent folder
 * MujocoShared - Shared files
   * Scripts/MujocoAgent.cs - Base Agent class for Mujoco implementations
   * Scripts/MujocoSpawner.cs - Class for creating a Unity game object from a MuJoCo.xml file
   * Scripts/MujocoJoint.cs - Model for mapping MuJoCo joints to Unity
   * Scripts/MujocoSensor.cs - Model for mapping MuJoCo sensors to Unity
   * Scripts/MujocoHelper.cs - Helper functions for MujocoSpawner.cs
   * Scripts/HandleOverlap.cs - helper script to for detecting overlapping MuJoCo elements.
   * Scripts/ProceduralCapsule.cs - Creates a Unity capsule which matches MuJoCo capsule
   * Scripts/SendOnCollisionTrigger.cs - class for sending collisions to MujocoAgent.cs
   * Scripts/SensorBehavior.cs - behavior class for sensors
   * Scripts/SmoothFollow.cs - camera script
 * DeepMindReferenceXml - Mujoco xml files used in DeepMind research (source)
 * OpenAIReferenceXml - Mujoco xml files used in OpenAI research (source)
 * DeepMindReferenceXml - Mujoco xml files used in DeepMind research (source)
 * DeepMindHopper - Folder for reproducing DeepMindHopper 
 * OpenAIHopper - Folder for reproducing OpenAIHopper 
 * etc

### Tuning params / Magic numbers
* xxNamexx\Prefab\xxNamexx -> MujocoSpawner.Force2D = set to True when implementing a 2d model (hopper, walker)
* xxNamexx\Prefab\xxNamexx -> MujocoSpawner.DefaultDesity:
  * 1000 = default (= same as Mujoco)
  * Note: maybe overriden within a .xml script
* xxNamexx\Prefab\xxNamexx -> MujocoSpawner.MotorScale = Magic number for tuning (scaler applied to all motors)
  * 1 = default () 
  * 0.1 used by OpenAIAnt
  * 0.5 used by OpenAIWalker
  * 1.5 used by DeepMindHopper, DeepMindWalker
  
* xxNamexx\Prefab\xxNamexx -> xxAgentScript.MaxStep / DecisionFrequency: 
  * 5000,5: OpenAIAnt, OpenAIHumanoid, DeepMindHumanoid
  * 4000,4: OpenAIHopper, OpenAIWalker, DeepMindHopper, DeepMindWalker
  * Note: all params taken from OpenAI.Gym





