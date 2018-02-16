# Making a new Learning Environment

This tutorial walks through the process of creating a Unity Environment. A Unity Environment is an application built using the Unity Engine which can be used to train Reinforcement Learning agents.

## Setting up ML Agents

To use ML Agents, follow the [installation instructions](Installation.md) and then do the following in your Unity project:

1. Create an environment for your agents to live in. An environment can range from a simple physical simulation containing a few objects to an entire game or ecosystem.
2. Implement an Academy subclass and add it to a GameObject in the Unity scene containing the environment. This GameObject will serve as the parent for any Brain objects in the scene. Your Academy class can implement a few optional methods to update the scene independently of any agents. For example, you can add, move, or delete agents and other entities in the environment.
3. Add one or more Brain objects to the scene as children of the Academy.
4. Implement your Agent subclasses. An Agent subclass defines the code an agent uses to observe its environment, to carry out assigned actions, and to calculate the rewards used for reinforcement training. You can also implement optional methods to reset the agent when it has finished or failed its task.
5. Add your Agent subclasses to appropriate GameObjects, typically, the object in the scene that represents the agent in the simulation. Each Agent object must be assigned a Brain object.
6. If training, set the Brain type to External and [run the training process](Training-with-PPO.md).  


**Note:** If you are unfamiliar with Unity, refer to [Learning th interface](https://docs.unity3d.com/Manual/LearningtheInterface.html) in the Unity Manual if an Editor task isn't explained sufficiently in this tutorial.

## Set Up the Unity Project

The first task to accomplish is simply creating a new Unity project and importing the ML Agents assets into it:

1. Launch the Unity Editor and create a new project named "RollerBall".
2. If necessary, clone the ML Agents repository with the `git` command:

    git clone git@github.com:Unity-Technologies/ml-agents.git
    
3. In a file system window, navigate to the folder containing your cloned ML Agents repository. 

4. Drag the `ML-Agents` folder from `unity-environments/Assets` to the Unity Editor Project window.

Your Unity **Project** window should contain the following assets:

![Project window](images/mlagents-NewProject.png)

## Create the Environment:

Next, we will create a very simple scene to act as our ML Agents environment. The "physical" components of the environment include a Plane to act as the floor for the agent to move around on, a Cube to act as the goal or target for the agent to seek, and a Sphere to represent the agent itself. 

**Create the floor plane:**

1. Right click in Hierarchy window, select 3D Object > Plane.
2. Name the GameObject "Floor."
3. Select Plane to view its properties in the Inspector window.
4. Set Transform to Position = (0,0,0), Rotation = (0,0,0), Scale = (1,1,1).
5. On the Plane's Mesh Renderer, expand the Materials property and change the default-material to *floor*.

(To set a new material, click the small circle icon next to the current material name. This opens the **Object Picker** dialog so that you can choose the a different material from the list of all materials currently in the project.)

![The Floor in the Inspector window](images/mlagent-NewTutFloor.png)

**Add the Target Cube**

1. Right click in Hierarchy window, select 3D Object > Cube.
2. Name the GameObject "Target"
3. Select Target to view its properties in the Inspector window.
4. Set Transform to Position = (3,0.5,3), Rotation = (0,0,0), Scale = (1,1,1).
5. On the Cube's Mesh Renderer, expand the Materials property and change the default-material to *block*.

![The Target Cube in the Inspector window](images/mlagent-NewTutBlock.png)

**Add the Agent Sphere**

1. Right click in Hierarchy window, select 3D Object > Sphere.
2. Name the GameObject "RollerAgent"
3. Select Target to view its properties in the Inspector window.
4. Set Transform to Position = (0,0.5,0), Rotation = (0,0,0), Scale = (1,1,1).
5. On the Sphere's Mesh Renderer, expand the Materials property and change the default-material to *checker 1*.
6. Click **Add Component**.
7. Add the Physics/Rigidbody component to the Sphere. (Adding a Rigidbody ) 

![The Target Cube in the Inspector window](images/mlagent-NewTutSphere.png)

Note that we will create an Agent subclass to add to this GameObject as a component later in the tutorial.

**Add Empty GameObjects to Hold the Academy and Brain**

1. Right click in Hierarchy window, select Create Empty.
2. Name the GameObject "Academy"
3. Right-click on the Academy GameObject and select Create Empty.
4. Name this child of the Academy, "Brain".

![The scene hierarchy](images/mlagent-NewTutHierarchy.png)

You can adjust the camera angles to give a bettwer view of the scene at runtime. The next steps will be to create and add the ML Agent components.

## Implement an Academy

The Academy object coordinates the ML Agents in the scene and drives the decision-making portion of the simulation loop. Every ML Agent scene needs one Academy instance. Since the base Academy classis abstract, you must make your own subclass even if you don't need to use any of the methods for a particular environment.

First, add a New Script component to the Academy GameObject created earlier: 
1. Select the Academy GameObject to view it in the Inspector window.
2. Click **Add Component**.
3. Click **New Script** in the list of components (at the bottom).
4. Name the script "RollerAcademy".
5. Click **Create and Add**.

Next, edit the new `RollerAcademy` script:
1. In the Unity Project window, double-click the `RollerAcademy` script to open it in your code editor. (By default new scripts are placed directly in the **Assets** folder.)
2. In the editor, change the base class from `MonoBehaviour` to `Academy`.
3. Delete the `Start()` and `Update()` methods that were added by default.

In such a basic scene, we don't need the Academy to initialize, reset, or otherwise control any objects in the environment so we have the simplest possible Academy implementation:
 
    using UnityEngine;

    public class RollerAcademy : Academy {}
     

The default settings for the Academy properties are also fine for this environment, so we don't need to change anything for the RollerAcademy component in the Inspector window.

![The Academy properties](images/mlagent-NewTutAcademy.png)

## Add a Brain

To Add a Brain:

1. Right-click the Academy GameObject in the Hierarchy window and choose *Create Empty* to add a child GameObject.
2. Name the new GameObject, "Brain".
3. Select the Brain to show its properties in the Inspector window.
4. Click **Add Component**.
5. Select the **Scripts/Brain** component to add it to the GameObject.

We will come back to the Brain properties later, but leave the Brain Type as **Player** for now.

![The Brain default properties](images/mlagent-NewTutBrain.png)

## Implement an Agent

To create the Agent:

1. Select the RollerAgent GameObject to view it in the Inspector window.
2. Click **Add Component**.
3. Click **New Script** in the list of components (at the bottom).
4. Name the script "RollerAgent".
5. Click **Create and Add**.

Then, edit the new `RollerAgent` script:

1. In the Unity Project window, double-click the `RollerAgent` script to open it in your code editor. 
2. In the editor, change the base class from `MonoBehaviour` to `Agent`.
3. Delete the `Update()` method, but we will use the `Start()` function, so leave it alone for now.

So far, these are the basic steps that you would use to add ML Agents to any Unity project. Next, we will add the logic that will let our agent learn to roll to the cube.

In this simple scenario, we don't need the Academy object do do anything special. If we wanted to change the environment, for example change the size of the floor or add or remove agents or other objects before or during the simulation, we could implement the appropriate methods in the Academy. Instead, we will have the Agent do all the work of resetting itself and the target when it succeeds or falls trying. 

When agent reaches its target, it marks itself done and its agent reset function moves the target to a random location. In addition, if the agent rolls off the platform, the reset function puts it back onto the floor.

To move the target GameObject, we need a reference to its Transform (which stores a GameObject's position, orientation and scale in the 3D world). To get this reference, add a public field of type `Transform` to the RollerAgent class.  Public fields of a component in Unity get displayed in the Inspector window, allowing you to choose which GameObject to use as the target in the Unity Editor. Our `AgentReset()` function looks like:

    public Transform Target;
    public override void AgentReset()
    {
        if(this.transform.position.y < -1.0){ //The agent fell
            this.transform.position = Vector3.zero;
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
        }
        else { //move the target to a new spot
            Target.position = new Vector3(Random.value * 8 - 4,
                                          0.5f,
                                          Random.value * 8 - 4);
        }
    }

Next let's implement the Agent.CollectState() function. The Agent sends the information we collect to the Brain, which uses it to make a decision. When you train the agent using the PPO training algorithm, the data is fed into a neural network as a feature vector. 

We want the  position as well as the position of the target. In general, it is better to use the relative position of other objects rather than the absolute position for more generalizable training.  That means the state observation contains 5 values and we need to use the continuous state space type when we get around to setting the Brain properties:

    //Observations
    List<float> observation = new List<float>();
    public override List<float> CollectState()
    {
        //Calculate relative position and angle:
        Vector3 relativePosition = this.transform.position - Target.position;
        float angleToTarget = Vector3.Angle(this.transform.forward, relativePosition);
                                     
        observation.Clear();
        observation.Add(this.transform.position.x/5); //Dividing by 5 to normalize positions to range [-1,1]
        observation.Add(this.transform.position.z/5);
        observation.Add(relativePosition.x/5);
        observation.Add(relativePosition.z/5);
        observation.Add(angleToTarget);

        return observation;
    }

The next bit we need to program for the Agent is the action function. We will use the continuous action space, so we need two action control values one to specify how much force to apply along the x axis, and one to apply a force in the z direction. . 

Finally we need to program the reward system so that we can train the Agent to roll to the targets and avoid falling off the platform. The main rewards are:

* Reaching the target = +1
* Falling from the platform = -1


Rewards are also assigned in the AgentStep() function, so the final version of the function looks like:

    public override void AgentStep(float[] act)
    {
        float distanceToTarget = Vector3.Distance(this.transform.position, Target.position);
        if( distanceToTarget < 1.7f){
            this.done = true;
            reward += 1.0f;
        }
        if(this.transform.position.y < -1.0){
            this.done = true;
            reward += -1.0f;
        }
        if(act[0] > 0){
            reward += 0.01f;
        }
        reward += -0.1f;

        //Action Size = 2
        chanController.AgentVerticalInput = Mathf.Clamp(act[0], -1, 1); 
        chanController.AgentHorizontalInput = Mathf.Clamp(act[1], -1, 1);
    }
    


Now, let's setup the brain so we can test the Agent using direct keyboard control:

Select the Brain GameObject so that the Brain shows in the Inspector window. Set these properties:

State Size = 5
Action Size = 2
Action Space Type = Continuous
State Space Type = Continuous
Brain Type = Player

Next let's hook up the keyboard controls. Click the triangle icon next to Continuous Player Actions to reveal the Size field. Set size to 4. Although we only have two action variables, we will use one key to specify positive values (i.e. forward) and one to specify negative values (i.e. backward) for each axis.


The Index value corresponds to the index of the action (act[]) array passed to AgentStep() function. Value is assigned to act[Index] when Key is pressed.

Some final changes:

Select RollerAgent so that you can see the  component in the Inspector.
Drag the Brain GameObject to the RollerAgent Brain field.

## Test the Environment

Drag the Target GameObject to the RollerAgent Target field.


Press Play to run the scene and use the WASD keys to test.

Now we can train the Agent. To get ready, we have to change the Brain Type from Player to **External**. From there the process is the same as described in Getting Started with the 3D Balance Ball Environment.

