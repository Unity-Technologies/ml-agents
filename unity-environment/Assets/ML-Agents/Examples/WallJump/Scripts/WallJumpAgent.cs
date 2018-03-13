//Put this script on your blue cube.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class WallJumpAgent : Agent
{
    // Depending on this value, the wall will have different height
    int configuration;
    // Brain to use when no wall is present
    public Brain noWallBrain;
    // Brain to use when a jumpable wall is present
    public Brain smallWallBrain;
    // Brain to use when a wall requiring a block to jump over is present
    public Brain bigWallBrain;

    public GameObject ground;
    public GameObject spawnArea;
    Bounds spawnAreaBounds;


    public GameObject goal;
    public GameObject shortBlock;
    public GameObject wall;
    Rigidbody shortBlockRB;
    Rigidbody agentRB;
    Material groundMaterial;
    Renderer groundRenderer;
    WallJumpAcademy academy;
    RayPerception rayPer;

    public float jumpingTime;
    public float jumpTime;
    // This is a downward force applied when falling to make jumps look
    // less floaty
    public float fallingForce;
    // Use to check the coliding objects
    public Collider[] hitGroundColliders = new Collider[3];
    Vector3 jumpTargetPos;
    Vector3 jumpStartingPos;

    string[] detectableObjects;

    public override void InitializeAgent()
    {
        academy = FindObjectOfType<WallJumpAcademy>();
        rayPer = GetComponent<RayPerception>();
        configuration = Random.Range(0, 5);
        detectableObjects = new string[] { "wall", "goal", "block" };

        agentRB = GetComponent<Rigidbody>();
        shortBlockRB = shortBlock.GetComponent<Rigidbody>();
        spawnAreaBounds = spawnArea.GetComponent<Collider>().bounds;
        groundRenderer = ground.GetComponent<Renderer>();
        groundMaterial = groundRenderer.material;

        spawnArea.SetActive(false);
    }


    // Begin the jump sequence
    public void Jump()
    {

        jumpingTime = 0.2f;
        jumpStartingPos = agentRB.position;
    }

    /// <summary>
    /// Does the ground check.
    /// </summary>
    /// <returns><c>true</c>, if the agent is on the ground, 
    /// <c>false</c> otherwise.</returns>
    /// <param name="boxWidth">The width of the box used to perform 
    /// the ground check. </param>
    public bool DoGroundCheck(float boxWidth)
    {
        hitGroundColliders = new Collider[3];
        Physics.OverlapBoxNonAlloc(
            gameObject.transform.position + new Vector3(0, -0.05f, 0),
                            new Vector3(boxWidth / 2f, 0.5f, boxWidth / 2f),
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


    /// <summary>
    /// Moves  a rigidbody towards a position smoothly.
    /// </summary>
    /// <param name="targetPos">Target position.</param>
    /// <param name="rb">The rigidbody to be moved.</param>
    /// <param name="targetVel">The velocity to target during the
    ///  motion.</param>
    /// <param name="maxVel">The maximum velocity posible.</param>
    void MoveTowards(
        Vector3 targetPos, Rigidbody rb, float targetVel, float maxVel)
    {
        Vector3 moveToPos = targetPos - rb.worldCenterOfMass;
        Vector3 velocityTarget = moveToPos * targetVel * Time.fixedDeltaTime;
        if (float.IsNaN(velocityTarget.x) == false)
        {
            rb.velocity = Vector3.MoveTowards(
                rb.velocity, velocityTarget, maxVel);
        }
    }

    public override void CollectObservations()
    {
        float rayDistance = 20f;
        float[] rayAngles = { 0f, 45f, 90f, 135f, 180f, 110f, 70f };
        AddVectorObs(rayPer.Perceive(
            rayDistance, rayAngles, detectableObjects, 0f, 0f));
        AddVectorObs(rayPer.Perceive(
            rayDistance, rayAngles, detectableObjects, 2.5f, 2.5f));
        Vector3 agentPos = agentRB.position - ground.transform.position;

        AddVectorObs(agentPos / 20f);
        AddVectorObs(DoGroundCheck(0.4f) ? 1 : 0);


    }

    /// <summary>
    /// Gets a random spawn position in the spawningArea.
    /// </summary>
    /// <returns>The random spawn position.</returns>
    public Vector3 GetRandomSpawnPos()
    {
        Vector3 randomSpawnPos = Vector3.zero;
        float randomPosX = Random.Range(-spawnAreaBounds.extents.x,
                                        spawnAreaBounds.extents.x);
        float randomPosZ = Random.Range(-spawnAreaBounds.extents.z,
                                        spawnAreaBounds.extents.z);

        randomSpawnPos = spawnArea.transform.position +
                                  new Vector3(randomPosX, 0.45f, randomPosZ);
        return randomSpawnPos;
    }

    /// <summary>
    /// Chenges the color of the ground for a moment
    /// </summary>
    /// <returns>The Enumerator to be used in a Coroutine</returns>
    /// <param name="mat">The material to be swaped.</param>
    /// <param name="time">The time the material will remain.</param>
    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        groundRenderer.material = mat;
        yield return new WaitForSeconds(time); //wait for 2 sec
        groundRenderer.material = groundMaterial;
    }


    public void MoveAgent(float[] act)
    {

        AddReward(-0.0005f);
        bool smallGrounded = DoGroundCheck(0.4f);
        bool largeGrounded = DoGroundCheck(1.0f);

        Vector3 dirToGo = Vector3.zero;
        Vector3 rotateDir = Vector3.zero;

        int action = Mathf.FloorToInt(act[0]);
        switch (action)
        {
            case 0:
                dirToGo = transform.forward * 1f * (largeGrounded ? 1f : 0.5f);
                break;
            case 1:
                dirToGo = transform.forward * -1f * (largeGrounded ? 1f : 0.5f);
                break;
            case 2:

                rotateDir = transform.up * -1f;
                break;
            case 3:
                rotateDir = transform.up * 1f;
                break;
            case 4:
                dirToGo = transform.right * -0.6f * (largeGrounded ? 1f : 0.5f);
                break;
            case 5:
                dirToGo = transform.right * 0.6f * (largeGrounded ? 1f : 0.5f);
                break;
            case 6:
                if ((jumpingTime <= 0f) && smallGrounded)
                {
                    Jump();
                }
                break;
        }

        transform.Rotate(rotateDir, Time.fixedDeltaTime * 300f);
        agentRB.AddForce(dirToGo * academy.agentRunSpeed,
                         ForceMode.VelocityChange);

        if (jumpingTime > 0f)
        {
            jumpTargetPos =
            new Vector3(agentRB.position.x,
                        jumpStartingPos.y + academy.agentJumpHeight,
                        agentRB.position.z) + dirToGo;
            MoveTowards(jumpTargetPos, agentRB, academy.agentJumpVelocity,
                        academy.agentJumpVelocityMaxChange);

        }

        if (!(jumpingTime > 0f) && !largeGrounded)
        {
            agentRB.AddForce(
            Vector3.down * fallingForce, ForceMode.Acceleration);
        }
        jumpingTime -= Time.fixedDeltaTime;
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        MoveAgent(vectorAction);
        if ((!Physics.Raycast(agentRB.position, Vector3.down, 20))
            || (!Physics.Raycast(shortBlockRB.position, Vector3.down, 20)))
        {
            Done();
            SetReward(-1f);
            ResetBlock(shortBlockRB);
            StartCoroutine(
                GoalScoredSwapGroundMaterial(academy.failMaterial, .5f));
        }
    }

    // Detect when the agent hits the goal
    void OnTriggerEnter(Collider col)
    {
        if (col.gameObject.CompareTag("goal") && DoGroundCheck(0.4f))
        {
            SetReward(1f);
            Done();
            StartCoroutine(
                GoalScoredSwapGroundMaterial(academy.goalScoredMaterial, 2));
        }
    }


    //Reset the orange block position
    void ResetBlock(Rigidbody blockRB)
    {
        blockRB.transform.position = GetRandomSpawnPos();
        blockRB.velocity = Vector3.zero;
        blockRB.angularVelocity = Vector3.zero;
    }

    public override void AgentReset()
    {
        ResetBlock(shortBlockRB);
        transform.localPosition = new Vector3(
            18 * (Random.value - 0.5f), 1, -12);
        configuration = Random.Range(0, 5);
        agentRB.velocity = default(Vector3);

    }

    private void FixedUpdate()
    {
        if (configuration != -1)
        {
            ConfigureAgent(configuration);
            configuration = -1;
        }
    }

    /// <summary>
    /// Configures the agent. Given an integer config, the wall will have
    /// different height and a different brain will be assigned to the agent.
    /// </summary>
    /// <param name="config">Config. 
    /// If 0 : No wall and noWallBrain.
    /// If 1:  Small wall and smallWallBrain.
    /// Other : Tall wall and BigWallBrain. </param>
    void ConfigureAgent(int config)
    {
        if (config == 0)
        {
            wall.transform.localScale = new Vector3(
                wall.transform.localScale.x,
                academy.resetParameters["no_wall_height"],
                wall.transform.localScale.z);
            GiveBrain(noWallBrain);
        }
        else if (config == 1)
        {
            wall.transform.localScale = new Vector3(
                wall.transform.localScale.x,
                academy.resetParameters["small_wall_height"],
                wall.transform.localScale.z);
            GiveBrain(smallWallBrain);
        }
        else
        {
            float height =
                academy.resetParameters["big_wall_min_height"] +
                Random.value * (academy.resetParameters["big_wall_max_height"] -
                academy.resetParameters["big_wall_min_height"]);
            wall.transform.localScale = new Vector3(
                wall.transform.localScale.x,
                height,
                wall.transform.localScale.z);
            GiveBrain(bigWallBrain);
        }
    }
}

