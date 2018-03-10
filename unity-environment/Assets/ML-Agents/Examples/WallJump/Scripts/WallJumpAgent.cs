//Put this script on your blue cube.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class WallJumpAgent : Agent
{
    int configuration;
    public Brain noWallBrain;
    public Brain smallWallBrain;
    public Brain bigWallBrain;

    public GameObject ground; //ground game object. we will use the area bounds to spawn the blocks
    public GameObject spawnArea; //ground game object. we will use the area bounds to spawn the blocks
    public bool visualizeSpawnArea;
    public Bounds spawnAreaBounds; //the bounds of the pushblock area
    public Bounds areaBounds; //the bounds of the pushblock area

    public GameObject goal; //goal to push the block to
    public GameObject shortBlock; //the orange block we are going to be pushing
    //public GameObject mediumBlock; //the orange block we are going to be pushing
    //public GameObject tallBlock; //the orange block we are going to be pushing
    public GameObject wall; //
    Rigidbody shortBlockRB;  //cached on initialization
                             //Rigidbody mediumBlockRB;  //cached on initialization
                             //Rigidbody tallBlockRB;  //cached on initialization
    Rigidbody agentRB;  //cached on initialization
    Material groundMaterial; //cached on Awake()
    Renderer groundRenderer;
    //Vector3 goalStartingPos;
    WallJumpAcademy academy;


    //JUMPING STUFF
    public float jumpingTime;
    public float jumpTime;
    public float fallingForce; //this is a downward force applied when falling to make jumps look less floaty
    public Collider[] hitGroundColliders = new Collider[3]; //used for groundchecks
    public bool visualizeGroundCheckSphere;
    //public bool grounded;
    public bool performingGroundCheck;
    public float groundCheckRadius; //the radius from transform.position to check
    public Vector3 groundCheckOffset; //offset groundcheck pos. useful for tweaking groundcheck box
    public float groundCheckFrequency; //perform a groundcheck every x sec. ex: .5 will do a groundcheck every .5 sec.
    Vector3 jumpTargetPos; //target this position during jump. it will be 
    Vector3 jumpStartingPos; //target this position during jump. it will be 

    string[] detectableObjects;

    void Awake()
    {
        academy = FindObjectOfType<WallJumpAcademy>();
    }

    public override void InitializeAgent()
    {
        configuration = Random.Range(0, 5);
        detectableObjects = new string[] { "wall", "goal", "block" };
        //StartGroundCheck();

        agentRB = GetComponent<Rigidbody>(); //cache the agent rigidbody
        shortBlockRB = shortBlock.GetComponent<Rigidbody>(); //cache the block rigidbody
        areaBounds = ground.GetComponent<Collider>().bounds; //get the ground's bounds
        spawnAreaBounds = spawnArea.GetComponent<Collider>().bounds; //get the ground's bounds
        groundRenderer = ground.GetComponent<Renderer>(); //get the ground renderer so we can change the material when a goal is scored
        groundMaterial = groundRenderer.material; //starting material

        spawnArea.SetActive(false);
    }


    //put agent into the jumping state for the specified jumpTime
    public void Jump()
    {

        jumpingTime = 0.2f;
        jumpStartingPos = agentRB.position;
        // jumpTargetPos = agentRB.position + Vector3.up * jumpHeight;
        //yield return new WaitForSeconds(jumpTime);
        //jumping = false;
        // StartCoroutine(Falling());//should be falling now
    }


    //GROUND CHECK STUFF (Used for jumping)
    public void StartGroundCheck()
    {
        if (!IsInvoking("DoGroundCheck"))
        {
            InvokeRepeating("DoGroundCheck", 0, groundCheckFrequency);
            performingGroundCheck = true;
        }
    }

    public void StopGroundCheck()
    {
        CancelInvoke("DoGroundCheck"); //stop doing ground check;
        performingGroundCheck = false;
    }

    // GROUND CHECK
    //public void DoGroundCheck()
    //{
    //hitGroundColliders = new Collider[3];
    //Physics.OverlapBoxNonAlloc(gameObject.transform.position + new Vector3(0, -0.05f, 0),
    //                           new Vector3(0.2f,0.5f,0.2f),
    //                                   hitGroundColliders,
    //                           gameObject.transform.rotation);
    //grounded = false;
    //foreach (Collider col in hitGroundColliders)
    //{

    //    if (col != null && col.transform != this.transform && 
    //        (col.CompareTag("walkableSurface") ||
    //        col.CompareTag("block") ||
    //         col.CompareTag("wall")))
    //    {
    //        grounded = true; //then we're grounded
    //        break;
    //    }
    //}

    //Vector3 posToUse = agentRB.position + groundCheckOffset;
    //int numberGroundCollidersHit = Physics.OverlapSphereNonAlloc(posToUse, groundCheckRadius, hitGroundColliders); //chose .6 radius because this should make a sphere a little bit bigger than our cube that is a scale of 1 unit. sphere will be 1.2 units. 
    //if (numberGroundCollidersHit > 1 )
    //{
    //	grounded = false;
    //	foreach(Collider col in hitGroundColliders)
    //	{
    //		if(col != null && col.transform != this.transform && col.CompareTag("walkableSurface"))
    //		{
    //			//build a random position to use for a groundcheck raycast
    //			Vector3 randomRaycastPos = agentRB.position;
    //			randomRaycastPos += agentRB.transform.forward * Random.Range(-.5f, .5f); // random forward/back
    //			randomRaycastPos += agentRB.transform.right * Random.Range(-.5f, .5f); // plus a random left/right
    //			if (Physics.Raycast(randomRaycastPos, Vector3.down, .8f)) //if we hit
    //			{
    //				grounded = true; //then we're grounded
    //				break;
    //			}
    //		}
    //	}
    //}
    //else
    //{
    //	grounded = false;
    //}

    // GROUND CHECK
    public bool DoSmallGroundCheck()
    {
        hitGroundColliders = new Collider[3];
        Physics.OverlapBoxNonAlloc(gameObject.transform.position + new Vector3(0, -0.05f, 0),
                                   new Vector3(0.2f, 0.5f, 0.2f),
                                           hitGroundColliders,
                                   gameObject.transform.rotation);
        bool grounded = false;
        foreach (Collider col in hitGroundColliders)
        {

            if (col != null && col.transform != this.transform &&
                (col.CompareTag("walkableSurface") ||
                col.CompareTag("block") ||
                 col.CompareTag("wall")))
            {
                grounded = true; //then we're grounded
                break;
            }
        }
        return grounded;
    }

    // GROUND CHECK
    public bool DoLargeGroundCheck()
    {
        hitGroundColliders = new Collider[3];
        Physics.OverlapBoxNonAlloc(gameObject.transform.position + new Vector3(0, -0.05f, 0),
                                   new Vector3(0.5f, 0.5f, 0.5f),
                                           hitGroundColliders,
                                   gameObject.transform.rotation);
        bool grounded = false;
        foreach (Collider col in hitGroundColliders)
        {

            if (col != null && col.transform != this.transform &&
                (col.CompareTag("walkableSurface") ||
                col.CompareTag("block") ||
                 col.CompareTag("wall")))
            {
                grounded = true; //then we're grounded
                break;
            }
        }
        return grounded;
    }



    //Draw the Box Overlap as a gizmo to show where it currently is testing. Click the Gizmos button to see this
    //void OnDrawGizmos()
    //{
    //    Gizmos.color = Color.red;
    //    //Check that it is being run in Play Mode, so it doesn't try to draw this in Editor mode
    //    //Draw a cube where the OverlapBox is (positioned where your GameObject is as well as a size)
    //    Gizmos.DrawWireCube(agentRB.position + new Vector3(0, -0.05f, 0),
    //                               agentRB.transform.localScale);
    //}

	//moves a rigidbody towards a position with a smooth controlled movement.
	void MoveTowards(Vector3 targetPos, Rigidbody rb, float targetVel, float maxVel)
	{
		Vector3 moveToPos = targetPos - rb.worldCenterOfMass;  //cube needs to go to the standard Pos
		//Vector3 velocityTarget = moveToPos * targetVel * Time.deltaTime; //not sure of the logic here, but it modifies velTarget
        /*Well does that fuck shit up ???? */
        Vector3 velocityTarget = moveToPos * targetVel * Time.fixedDeltaTime;
		if (float.IsNaN(velocityTarget.x) == false) //sanity check. if the velocity is NaN that means it's going way too fast. this check isn't needed for slow moving objs
		{
			rb.velocity = Vector3.MoveTowards(rb.velocity, velocityTarget, maxVel);
		}
	}

    public void RayPerception(float rayDistance,
                             float[] rayAngles, string[] detectableObjects,
                              float startHeight, float endHeight)
    {
        foreach (float angle in rayAngles)
        {
            float noise = 0f;
            float noisyAngle = angle + Random.Range(-noise, noise);
            Vector3 position = transform.TransformDirection(
                GiveCatersian(rayDistance, noisyAngle));
            position.y = endHeight;
            Debug.DrawRay(transform.position + new Vector3(0f, startHeight, 0f),
                          position, Color.red, 0.1f, true);
            RaycastHit hit;
            float[] subList = new float[detectableObjects.Length + 2];
            if (Physics.SphereCast(transform.position +
                                   new Vector3(0f, startHeight, 0f), 1.0f,
                                   position, out hit, rayDistance))
            {
                for (int i = 0; i < detectableObjects.Length; i++)
                {
                    if (hit.collider.gameObject.CompareTag(detectableObjects[i]))
                    {
                        subList[i] = 1;
                        subList[detectableObjects.Length + 1] = hit.distance / rayDistance;
                        break;
                    }
                }
            }
            else
            {
                subList[detectableObjects.Length] = 1f;
            }
            foreach (float f in subList)
                AddVectorObs(f);
        }
    }
    public Vector3 GiveCatersian(float radius, float angle)
    {
        float x = radius * Mathf.Cos(DegreeToRadian(angle));
        float z = radius * Mathf.Sin(DegreeToRadian(angle));
        return new Vector3(x, 1f, z);
    }

    public float DegreeToRadian(float degree)
    {
        return degree * Mathf.PI / 180f;
    }

	public override void CollectObservations()
	{
        float rayDistance = 20f;
        float[] rayAngles = { 0f, 45f, 90f, 135f, 180f, 110f, 70f };
        RayPerception(rayDistance, rayAngles, detectableObjects, 0f, 0f);
        RayPerception(rayDistance, rayAngles, detectableObjects, 2.5f, 2.5f);


  //      Vector3 goalPos = goal.transform.position - agentRB.position;  //pos of goal rel to agent
  //      Vector3 shortBlockPos = shortBlockRB.transform.position - ground.transform.position;  //pos of goal rel to agent
		Vector3 agentPos = agentRB.position - ground.transform.position;  //pos of agent rel to ground

        ////COLLECTIN STATES
        //AddVectorObs((shortBlockRB.transform.position - agentRB.position) / 20f);
        AddVectorObs(agentPos /20f);  //pos of agent rel to ground
        //AddVectorObs(goalPos / 20f);  //pos of goal rel to ground
        //AddVectorObs(shortBlockPos / 20f);  //pos of short block rel to ground
        //AddVectorObs(agentRB.velocity / 20f); //agent's vel
        //AddVectorObs(agentRB.transform.rotation.eulerAngles /180f - Vector3.one); //agent's rotation
        AddVectorObs(DoSmallGroundCheck() ? 1 : 0);


	}

	//use the ground's bounds to pick a random spawn pos
    public Vector3 GetRandomSpawnPos()
    {
        Vector3 randomSpawnPos = Vector3.zero;
        float randomPosX = Random.Range(-spawnAreaBounds.extents.x * academy.spawnAreaMarginMultiplier, spawnAreaBounds.extents.x * academy.spawnAreaMarginMultiplier);
        float randomPosZ = Random.Range(-spawnAreaBounds.extents.z * academy.spawnAreaMarginMultiplier, spawnAreaBounds.extents.z * academy.spawnAreaMarginMultiplier);
        randomSpawnPos = spawnArea.transform.position + new Vector3(randomPosX, 1.5f -1.05f, randomPosZ );
        return randomSpawnPos;
    }

	//swap ground material, wait time seconds, then swap back to the regular ground material.
	IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
	{
		groundRenderer.material = mat;
		yield return new WaitForSeconds(time); //wait for 2 sec
		groundRenderer.material = groundMaterial;
	}


	public void MoveAgent(float[] act) 
	{



        //AGENT ACTIONS
        // this is where we define the actions our agent can use...stuff like "go left", "go forward", "turn" ...etc.

        //Continuous control Vs. Discrete control

        //If we're using Continuous control you will need to change the Action
        if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
        {

            //AddReward(-0.0001f); //existential penalty

            //act[0] = Mathf.Clamp(act[0], -1, 1);
            //act[1] = Mathf.Clamp(act[1], -1, 1);
            //float speedX = 0;
            //float speedZ = 0;
            //if (act[0] != 0)
            //{
            //    speedX = grounded ? act[0] : act[0] / 2; //if we are in the air, our move speed should be a fraction of normal speed.
            //}
            //if (act[1] != 0)
            //{
            //    speedZ = grounded ? act[1] : act[1] / 2; //if we are in the air, our move speed should be a fraction of normal speed.
            //}

            //Vector3 directionX = Vector3.right * speedX;  //go left or right in world space
            //Vector3 directionZ = Vector3.forward * speedZ; //go forward or back in world space
            //Vector3 dirToGo = directionX + directionZ; //the dir we want to go

            //if (act[2] > 0 && !(jumpingTime > 0f) && grounded)
            //{
            //    //jump
            //    //AddReward(-0.0005f); //energy conservation penalty
            //    //StartCoroutine(Jump());
            //    Jump();
            //}

            //if (jumpingTime > 0f)
            //{

            //    //agentRB.AddForce(Vector3.up * 10, ForceMode.VelocityChange); 
            //    jumpTargetPos = new Vector3(agentRB.position.x, jumpStartingPos.y + academy.agentJumpHeight, agentRB.position.z) + dirToGo/2;// + transform.forward / 4;

            //    MoveTowards(jumpTargetPos, agentRB, academy.agentJumpVelocity, academy.agentJumpVelocityMaxChange);

            //}

            //if (!(jumpingTime > 0f) && !grounded) //add some downward force so it's not floaty
            //{
            //    agentRB.AddForce(Vector3.down * fallingForce, ForceMode.Acceleration);
            //}
            //jumpingTime -= Time.fixedDeltaTime;


            ////add force
            //agentRB.AddForce(dirToGo * academy.agentRunSpeed, ForceMode.VelocityChange); //GO
            //                                                                             //agentRB.velocity = dirToGo * academy.agentRunSpeed;
            //                                                                             //rotate the player forward
            //if (dirToGo != Vector3.zero)
            //{
            //    //agentRB.rotation = Quaternion.Lerp(agentRB.rotation, Quaternion.LookRotation(dirToGo), Time.deltaTime * academy.agentRotationSpeed);
            //    agentRB.rotation = Quaternion.Lerp(agentRB.rotation, Quaternion.LookRotation(dirToGo), Time.fixedDeltaTime * academy.agentRotationSpeed);
            //}
        }
        else
        {
            //float speedX = 0;
            //float speedZ = 0;
            //AddReward(-0.0005f); //existential penalty
            //int action = (int)(act[0]);
            //if (action == 1)
            //{
            //    speedX = grounded ? -1f : -0.5f ;
            //}
            //else if (action == 2)
            //{
            //    speedX = grounded ? 1f : 0.5f ;
            //}
            //else if (action == 3)
            //{
            //    speedZ = grounded ? 1f : 0.5f ; //if we are in the air, our move speed should be a fraction of normal speed.
            //}
            //else if (action == 4)
            //{
            //    speedZ = grounded ? -1f : -0.5f ; //if we are in the air, our move speed should be a fraction of normal speed.
            //}
            //else if ((action == 5) && (jumpingTime <= 0f) && grounded)
            //{
            //    //AddReward(-0.001f); //energy conservation penalty
            //    //StartCoroutine(Jump());
            //    Jump();
            //}



            //Vector3 directionX = Vector3.right * speedX;  //go left or right in world space
            //Vector3 directionZ = Vector3.forward * speedZ; //go forward or back in world space
            //Vector3 dirToGo = directionX + directionZ; //the dir we want to go
            //agentRB.AddForce(dirToGo * academy.agentRunSpeed, ForceMode.VelocityChange); //GO
            //agentRB.velocity = dirToGo * academy.agentRunSpeed;
            //rotate the player forward
            AddReward(-0.0005f);
            bool smallGrounded = DoSmallGroundCheck();
            bool largeGrounded = DoLargeGroundCheck();

            Vector3 dirToGo = Vector3.zero;
            Vector3 rotateDir = Vector3.zero;

            int action = Mathf.FloorToInt(act[0]);
            if (action == 0)
            {
                dirToGo = transform.forward * 1f * (largeGrounded ? 1f : 0.5f);
            }
            else if (action == 1)
            {
                dirToGo = transform.forward * -1f * (largeGrounded ? 1f : 0.5f);
            }
            else if (action == 2)
            {
                rotateDir = transform.up * -1f;
            }
            else if (action == 3)
            {
                rotateDir = transform.up * 1f;
            }
            else if (action == 4)
            {
                dirToGo = transform.right * -0.6f * (largeGrounded ? 1f : 0.5f);
            }
            else if (action == 5)
            {
                dirToGo = transform.right * 0.6f * (largeGrounded? 1f : 0.5f);
            }
            else if ((action == 6) && (jumpingTime <= 0f) && smallGrounded)
                {
                    //AddReward(-0.001f); //energy conservation penalty
                    //StartCoroutine(Jump());
                    Jump();
                }

            transform.Rotate(rotateDir, Time.fixedDeltaTime * 300f);
            agentRB.AddForce(dirToGo * academy.agentRunSpeed,
                             ForceMode.VelocityChange);

            if (jumpingTime > 0f)
            {

                //agentRB.AddForce(Vector3.up * 10, ForceMode.VelocityChange); 
                jumpTargetPos = new Vector3(agentRB.position.x, jumpStartingPos.y + academy.agentJumpHeight, agentRB.position.z)+dirToGo;// + transform.forward / 4;

                MoveTowards(jumpTargetPos, agentRB, academy.agentJumpVelocity, academy.agentJumpVelocityMaxChange);

            }

            if (!(jumpingTime > 0f) && !largeGrounded) //add some downward force so it's not floaty
            {
                agentRB.AddForce(Vector3.down * fallingForce, ForceMode.Acceleration);
            }
            jumpingTime -= Time.fixedDeltaTime;
            //if (dirToGo != Vector3.zero)
            //{
            //    //agentRB.rotation = Quaternion.Lerp(agentRB.rotation, Quaternion.LookRotation(dirToGo), Time.deltaTime * academy.agentRotationSpeed);
            //    agentRB.rotation = Quaternion.Lerp(agentRB.rotation, Quaternion.LookRotation(dirToGo), Time.fixedDeltaTime * academy.agentRotationSpeed);
            //}

        }
    }

	public override void AgentAction(float[] vectorAction, string textAction)
	{
        MoveAgent(vectorAction); //perform agent actions
		bool fail = false;  // did the agent or block get pushed off the edge?

		//debug
		if(visualizeSpawnArea && !spawnArea.activeInHierarchy)
		{
			spawnArea.SetActive(true);
		}


		if (!Physics.Raycast(agentRB.position, Vector3.down, 20)) //if the agent has gone over the edge, we done.
		{
			fail = true; //fell off bro
            SetReward(-1f); // BAD AGENT
			transform.position =  GetRandomSpawnPos();
            Done(); //if we mark an agent as done it will be reset automatically. AgentReset() will be called.
		}

		if (!Physics.Raycast(shortBlockRB.position, Vector3.down, 20)) //if the block has gone over the edge, we done.
		{
			fail = true; //fell off bro
            SetReward(-1f); // BAD AGENT
			ResetBlock(shortBlockRB); //reset block pos
            Done(); //if we mark an agent as done it will be reset automatically. AgentReset() will be called.
		}

		if (fail)
		{
			StartCoroutine(GoalScoredSwapGroundMaterial(academy.failMaterial, .5f)); //swap ground material to indicate fail
		}
	}

	// detect when we touch the goal
	void OnCollisionEnter(Collision col)
	{
		if(col.gameObject.CompareTag("goal")) //touched goal
		{
            SetReward(1f); //you get a point
            Done(); //if we mark an agent as done it will be reset automatically. AgentReset() will be called.
			StartCoroutine(GoalScoredSwapGroundMaterial(academy.goalScoredMaterial, 2)); //swap ground material for a bit to indicate we scored.
		}
	}
	
	
	//Reset the orange block position
	void ResetBlock(Rigidbody blockRB)
	{
		blockRB.transform.position = GetRandomSpawnPos(); //get a random pos
        blockRB.velocity = Vector3.zero; //reset vel back to zero
        blockRB.angularVelocity = Vector3.zero; //reset angVel back to zero
	}

	//In the editor, if "Reset On Done" is checked then AgentReset() will be called automatically anytime we mark done = true in an agent script.
	public override void AgentReset()
	{
		ResetBlock(shortBlockRB);
        //transform.position = GetRandomSpawnPos();
        transform.localPosition = new Vector3(9 * 2*( Random.value-0.5f), 1, -12);
        configuration = Random.Range(0, 5);

        //if (ground.transform.parent.position.x >= 0)
        //{
        //    configuration = Random.Range(0, 2);
        //}
        //if (ground.transform.parent.position.x < 0)
        //{
        //    configuration = Random.Range(2, 5);

        //}

	}

    private void FixedUpdate()
    {
        if (configuration != -1)
        {
            ConfigureAgent(configuration);
            configuration = -1;
        }
    }

    void ConfigureAgent(int config)
    {
        if (config == 0)
        {
            wall.transform.localScale = new Vector3(wall.transform.localScale.x, academy.resetParameters["no_wall_height"], wall.transform.localScale.z);
            GiveBrain(noWallBrain);
        }
        else if (config == 1)
        {
            wall.transform.localScale = new Vector3(wall.transform.localScale.x, academy.resetParameters["small_wall_height"], wall.transform.localScale.z);
            GiveBrain(smallWallBrain);
        }
        else
        {
            float height = academy.resetParameters["big_wall_min_height"] + Random.value * (academy.resetParameters["big_wall_max_height"] - academy.resetParameters["big_wall_min_height"]);
            wall.transform.localScale = new Vector3(wall.transform.localScale.x, height, wall.transform.localScale.z);
            GiveBrain(bigWallBrain);
        }
    }
}

