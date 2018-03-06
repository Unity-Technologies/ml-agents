//Put this script on your blue cube.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PushAgentBasic : Agent
{
    /// <summary>
    /// The ground. The bounds are used to spawn the elements.
    /// </summary>
	public GameObject ground;

    /// <summary>
    /// The area bounds.
    /// </summary>
	[HideInInspector]
	public Bounds areaBounds;

	PushBlockAcademy academy;
	
    /// <summary>
    /// The goal to push the block to.
    /// </summary>
    public GameObject goal;

    /// <summary>
    /// The block to be pushed to the goal.
    /// </summary>
    public GameObject block;

    /// <summary>
    /// The obstacle to be avoided while pushing the block.
    /// </summary>
    public GameObject obstacle;

    /// <summary>
    /// Detects when the block touches the goal.
    /// </summary>
	[HideInInspector]
	public GoalDetect goalDetect;

	Rigidbody blockRB;  //cached on initialization
	Rigidbody agentRB;  //cached on initialization
	Material groundMaterial; //cached on Awake()

    /// <summary>
    /// We will be changing the ground material based on success/failue
    /// </summary>
    Renderer groundRenderer;

	void Awake()
    {
        // There is one brain in the scene so this should find our brain.
		brain = FindObjectOfType<Brain>(); 
		academy = FindObjectOfType<PushBlockAcademy>(); //cache the academy
	}

    public override void InitializeAgent()
    {
		base.InitializeAgent();
		goalDetect = block.GetComponent<GoalDetect>();
		goalDetect.agent = this; 

        // Cache the agent rigidbody
		agentRB	= GetComponent<Rigidbody>(); 
        // Cache the block rigidbody
		blockRB	= block.GetComponent<Rigidbody>(); 
        // Get the ground's bounds
		areaBounds = ground.GetComponent<Collider>().bounds; 
        // Get the ground renderer so we can change the material when a goal is scored
		groundRenderer = ground.GetComponent<Renderer>(); 
        // Starting material
        groundMaterial = groundRenderer.material; 
    }

	public override void CollectObservations()
	{
  //      // Agent position relative to goal.
		//Vector3 agentPosRelToGoal = agentRB.position - goal.transform.position; 
  //      // Block position relative to goal.
		//Vector3 blockPosRelToGoal = blockRB.position - goal.transform.position;
  //      // Block position relative to obstacle.
		//Vector3 blockPosRelToObstacle = blockRB.position - obstacle.transform.position; 
  //      // Block position relative to agent.
		//Vector3 blockPosRelToAgent = blockRB.position - agentRB.position; 
  //      // Obstacle position relative to agent.
		//Vector3 obstaclePosRelToAgent = obstacle.transform.position - agentRB.position; 
		
        // Agent position relative to ground.
		Vector3 agentPos = agentRB.position - ground.transform.position; 
        // Goal position relative to ground.
		Vector3 goalPos = goal.transform.position - ground.transform.position;  
        // Block position relative to ground.
		Vector3 blockPos = blockRB.transform.position - ground.transform.position;  
        // Obstacle position relative to ground.
		Vector3 obstaclePos = obstacle.transform.position - ground.transform.position;

        // Collect Observations using helper function.
        Agent myAgent = GetComponent<Agent>();
        MLAgentsHelpers.CollectVector3State(myAgent, agentPos);
        MLAgentsHelpers.CollectVector3State(myAgent, goalPos);
        MLAgentsHelpers.CollectVector3State(myAgent, blockPos);
        MLAgentsHelpers.CollectVector3State(myAgent, obstaclePos);

        //MLAgentsHelpers.CollectVector3State(myAgent, agentPosRelToGoal);
        //MLAgentsHelpers.CollectVector3State(myAgent, blockPosRelToGoal);
        //MLAgentsHelpers.CollectVector3State(myAgent, blockPosRelToObstacle);
        //MLAgentsHelpers.CollectVector3State(myAgent, blockPosRelToAgent);
        //MLAgentsHelpers.CollectVector3State(myAgent, obstaclePosRelToAgent);

        // Add velocity of block and agent to observations.
        MLAgentsHelpers.CollectVector3State(myAgent, blockRB.velocity);
        MLAgentsHelpers.CollectVector3State(myAgent, agentRB.velocity);
	}

    /// <summary>
    /// Use the ground's bounds to pick a random spawn position.
    /// </summary>
    public Vector3 GetRandomSpawnPos(float spawnHeight)
    {
        bool foundNewSpawnLocation = false;
        Vector3 randomSpawnPos = Vector3.zero;
        while (foundNewSpawnLocation == false) 
        {
            float randomPosX = Random.Range(-areaBounds.extents.x * academy.spawnAreaMarginMultiplier,
                                areaBounds.extents.x * academy.spawnAreaMarginMultiplier);

            float randomPosZ = Random.Range(-areaBounds.extents.z * academy.spawnAreaMarginMultiplier,
                                            areaBounds.extents.z * academy.spawnAreaMarginMultiplier);
            randomSpawnPos = ground.transform.position + new Vector3(randomPosX, 1f, randomPosZ);
            if (Physics.CheckBox(randomSpawnPos, new Vector3(1.25f, 0.01f, 1.25f)) == false)
            {
                foundNewSpawnLocation = true;
            }
        }
        return randomSpawnPos;
    }

	/// <summary>
    /// Called when the agent moves the block into the goal.
    /// </summary>
	public void IScoredAGoal()
	{
        // We use a reward of 5.
        AddReward(5f);

        // By marking an agent as done AgentReset() will be called automatically.
        Done();

        // Swap ground material for a bit to indicate we scored.
		StartCoroutine(GoalScoredSwapGroundMaterial(academy.goalScoredMaterial, 1)); 
	}

	/// <summary>
    /// Called when block touches the obstacle.
    /// </summary>
	public void BlockTouchedObstacle()
	{
        AddReward(-1f);

        // By marking an agent as done AgentReset() will be called automatically.
        Done();

        // Swap ground material for a bit to indicate we scored.
		StartCoroutine(GoalScoredSwapGroundMaterial(academy.failMaterial, 1));

	}

	/// <summary>
    /// Swap ground material, wait time seconds, then swap back to the regular material.
    /// </summary>
    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
	{
		groundRenderer.material = mat;
		yield return new WaitForSeconds(time); // Wait for 2 sec
		groundRenderer.material = groundMaterial;
	}

    /// <summary>
    /// Moves the agent according to the selected action.
    /// </summary>
	public void MoveAgent(float[] act) 
	{
		// AGENT ACTIONS
		// Here we define the actions our agent can use, such as
        // "go left", "go forward", "turn", etc.

        if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
        {
			// Continuous control means that actions are on a sliding scale. 
			// In the brain we  define the number of axes we want to use here. 
            // In this example we need 2 axes to define:
			// Right/left movement (act[0])
			// Forward/back movement (act[1])
				
			// Example: Right/Left Movement. It is defined in this line:
			// Vector3 directionX = Vector3.right * Mathf.Clamp(act[0], -1f, 1f);

			// The neural network is setting the act[0] value.
			// If it chooses 1 then the agent will go right. 
			// If it chooses -1 the agent will go left. 
			// If it chooses .42 then it will go a little bit right
			// If it chooses -.8 then it will go left (well...80% left)
			
			// Energy Conservation Penalties
			// Give penalties based on how fast the agent chooses to go. 
            // The agent should only exert as much energy as necessary.
			// This is how animals work as well. 
            // i.e. You're probably not running in place at all times.

            // Larger the value, the less the penalty is.
			float energyConservPenaltyModifier = 10000;

            // The larger the movement, the greater the penalty given.
            AddReward(-Mathf.Abs(act[0])/energyConservPenaltyModifier); 
            AddReward(-Mathf.Abs(act[1])/energyConservPenaltyModifier);
			 
            // Move left or right in world space.
			Vector3 directionX = Vector3.right * Mathf.Clamp(act[0], -1f, 1f);  

            // Move forward or back in world space.
            Vector3 directionZ = Vector3.forward * Mathf.Clamp(act[1], -1f, 1f); 

            // Add directions together. This is the direction we want the agent
            // to move in.
        	Vector3 dirToGo = directionX + directionZ; 

            // Apply movement force!
			agentRB.AddForce(dirToGo * academy.agentRunSpeed, ForceMode.VelocityChange);

            // Rotate the agent appropriately.
			if(dirToGo != Vector3.zero)
			{
				//agentRB.rotation = Quaternion.Lerp(agentRB.rotation, 
                //                                   Quaternion.LookRotation(dirToGo), 
                //                                   Time.deltaTime * academy.agentRotationSpeed);
			}
        }
    }

    /// <summary>
    /// Called every step of the engine. Here the agent takes an action.
    /// </summary>
	public override void AgentAction(float[] act)
	{
        // Move the agent using the action.
        MoveAgent(act); 

        // Penalty given each step to encourage agent to finish task quickly.
        AddReward(-.00025f); 

        // Did the agent or block get pushed off the edge?
		bool fail = false;  

        // If the agent has gone over the edge, end the episode.
		if (!Physics.Raycast(agentRB.position, Vector3.down, 3))
		{
            // Fell off bro
			fail = true;

            // BAD AGENT
            AddReward(-1f);

            // If we mark an agent as done it will be reset automatically. 
            // AgentReset() will be called.
            Done();
		}

        // If the block has gone over the edge, end the episode.
		if (!Physics.Raycast(blockRB.position, Vector3.down, 3)) 
		{
            // Fell off bro
			fail = true; 

            // BAD AGENT
            AddReward(-1f); 

            // Reset block.
			ResetBlock();

            // If we mark an agent as done it will be reset automatically. 
            // AgentReset() will be called.
            Done();
		}

		if (fail)
		{
            // Swap ground material to indicate failure of the episode.
			StartCoroutine(GoalScoredSwapGroundMaterial(academy.failMaterial, 1f)); 
		}
	}
	
    /// <summary>
    /// Resets the block position and velocities.
    /// </summary>
	void ResetBlock()
	{
        // Get a random position for the block.
		block.transform.position = GetRandomSpawnPos(1.5f);

        // Reset block velocity back to zero.
        blockRB.velocity = Vector3.zero; 

        // Reset block angularVelocity back to zero.
        blockRB.angularVelocity = Vector3.zero;
	}


    /// <summary>
    /// In the editor, if "Reset On Done" is checked then AgentReset() will be 
    /// called automatically anytime we mark done = true in an agent script.
    /// </summary>
	public override void AgentReset()
	{
		ResetBlock();
        transform.position = GetRandomSpawnPos(1.5f);

        // Pick random spawn position with a y height of .35f. 
        // Height was chosen based on trial/error.
		goal.transform.position = GetRandomSpawnPos(.35f); 

        // Pick random spawn pos with a y height of .5f. 
        // Height was chosen based on trial/error.
		obstacle.transform.position = GetRandomSpawnPos(.5f); 
	}
}

