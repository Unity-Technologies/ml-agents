# Example Learning Environments

### About Example Environments
Unity ML Agents contains a set of example environments which demonstrate various features of the platform. In the coming months more will be added. We are also actively open to adding community contributed environments as examples, as long as they are small, simple, demonstrate a unique feature of the platform, and provide a unique non-trivial challenge to modern RL algorithms. Feel free to submit these environments with a Pull-Request explaining the nature of the environment and task. 

Environments are located in `unity-environment/ML-Agents/Examples`.

## Basic

* Set-up: A linear movement task where the agent must move left or right to rewarding states.
* Goal: Move to the most reward state.
* Agents: The environment contains one agent linked to a single brain.
* Agent Reward Function: 
    * +0.1 for arriving at suboptimal state.
    * +1.0 for arriving at optimal state.
* Brains: One brain with the following state/action space.
    * State space: (Discrete) One variable corresponding to current state.
    * Action space: (Discrete) Two possible actions (Move left, move right).
    * Observations: 0
* Reset Parameters: None

## 3DBall

![Balance Ball](../images/balance.png)

* Set-up: A balance-ball task, where the agent controls the platform. 
* Goal: The agent must balance the platform in order to keep the ball on it for as long as possible.
* Agents: The environment contains 12 agents of the same kind, all linked to a single brain.
* Agent Reward Function: 
    * +0.1 for every step the ball remains on the platform. 
    * -1.0 if the ball falls from the platform.
* Brains: One brain with the following state/action space.
    * State space: (Continuous) 8 variables corresponding to rotation of platform, and position, rotation, and velocity of ball.
    * Action space: (Continuous) Size of 2, with one value corresponding to X-rotation, and the other to Z-rotation.
    * Observations: 0
* Reset Parameters: None

## GridWorld

![GridWorld](../images/gridworld.png)

* Set-up: A version of the classic grid-world task. Scene contains agent, goal, and obstacles. 
* Goal: The agent must navigate the grid to the goal while avoiding the obstacles.
* Agents: The environment contains one agent linked to a single brain.
* Agent Reward Function: 
    * -0.01 for every step.
    * +1.0 if the agent navigates to the goal position of the grid (episode ends).
    * -1.0 if the agent navigates to an obstacle (episode ends).
* Brains: One brain with the following state/action space.
    * State space: (Continuous) 6 variables corresponding to position of agent and nearest goal and obstacle.
    * Action space: (Discrete) Size of 4, corresponding to movement in cardinal directions.
    * Observations: One corresponding to top-down view of GridWorld.
* Reset Parameters: Three, corresponding to grid size, number of obstacles, and number of goals.


## Tennis

![Tennis](../images/tennis.png)

* Set-up: Two-player game where agents control rackets to bounce ball over a net. 
* Goal: The agents must bounce ball between one another while not dropping or sending ball out of bounds.
* Agents: The environment contains two agent linked to a single brain.
* Agent Reward Function (independent): 
    * -0.1 To last agent to hit ball before going out of bounds or hitting ground/net (episode ends).
    * +0.1 To agent when hitting ball after ball was hit by the other agent. 
    * +0.1 To agent who didn't hit ball last when ball hits ground.
* Brains: One brain with the following state/action space.
    * State space: (Continuous) 8 variables corresponding to position and velocity of ball and racket.
    * Action space: (Discrete) Size of 4, corresponding to movement toward net, away from net, jumping, and no-movement.
    * Observations: None
* Reset Parameters: One, corresponding to size of ball.

## Area 

### Push Area

![Push](../images/push.png)

* Set-up: A platforming environment where the agent can push a block around.
* Goal: The agent must push the block to the goal.
* Agents: The environment contains one agent linked to a single brain.
* Agent Reward Function: 
    * -0.01 for every step.
    * +1.0 if the block touches the goal.
    * -1.0 if the agent falls off the platform.
* Brains: One brain with the following state/action space.
    * State space: (Continuous) 15 variables corresponding to position and velocities of agent, block, and goal.
    * Action space: (Discrete) Size of 6, corresponding to movement in cardinal directions, jumping, and no movement.
    * Observations: None.
* Reset Parameters: One, corresponding to number of steps in training. Used to adjust size of elements for Curriculum Learning.

### Wall Area

![Wall](../images/wall.png)

* Set-up: A platforming environment where the agent can jump over a wall.
* Goal: The agent must use the block to scale the wall and reach the goal.
* Agents: The environment contains one agent linked to a single brain.
* Agent Reward Function: 
    * -0.01 for every step.
    * +1.0 if the agent touches the goal.
    * -1.0 if the agent falls off the platform.
* Brains: One brain with the following state/action space.
    * State space: (Continuous) 16 variables corresponding to position and velocities of agent, block, and goal, plus the height of the wall.
    * Action space: (Discrete) Size of 6, corresponding to movement in cardinal directions, jumping, and no movement.
    * Observations: None.
* Reset Parameters: One, corresponding to number of steps in training. Used to adjust size of the wall for Curriculum Learning.

## Reacher

![Tennis](../images/reacher.png)

* Set-up: Double-jointed arm which can move to target locations.
* Goal: The agents must move it's hand to the goal location, and keep it there.
* Agents: The environment contains 32 agent linked to a single brain.
* Agent Reward Function (independent): 
    * +0.1 Each step agent's hand is in goal location.
* Brains: One brain with the following state/action space.
    * State space: (Continuous) 26 variables corresponding to position, rotation, velocity, and angular velocities of the two arm rigidbodies.
    * Action space: (Continuous) Size of 4, corresponding to torque applicable to two joints. 
    * Observations: None
* Reset Parameters: Two, corresponding to goal size, and goal movement speed.

## Crawler

![Crawler](../images/crawler.png)

* Set-up: A creature with 4 arms and 4 forearms.
* Goal: The agents must move its body along the x axis without falling.
* Agents: The environment contains 3 agent linked to a single brain.
* Agent Reward Function (independent): 
    * +1 times velocity in the x direction
    * -1 for falling.
    * -0.01 times the action squared
    * -0.05 times y position change
    * -0.05 times velocity in the z direction 
* Brains: One brain with the following state/action space.
    * State space: (Continuous) 117 variables corresponding to position, rotation, velocity, and angular velocities of each limb plus the acceleration and angular acceleration of the body.
    * Action space: (Continuous) Size of 12, corresponding to torque applicable to 12 joints. 
    * Observations: None
* Reset Parameters: None