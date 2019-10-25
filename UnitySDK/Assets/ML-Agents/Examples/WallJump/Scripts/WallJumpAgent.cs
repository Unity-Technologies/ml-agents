//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using MLAgents;

[RequireComponent(typeof(AgentCubeGroundCheck))] // Required for groundcheck
public class WallJumpAgent : Agent
{
    // Depending on this value, the wall will have different height
    int m_Configuration;
    // Brain to use when no wall is present
    public Brain noWallBrain;
    // Brain to use when a jumpable wall is present
    public Brain smallWallBrain;
    // Brain to use when a wall requiring a block to jump over is present
    public Brain bigWallBrain;

    public GameObject ground;
    public GameObject spawnArea;
    Bounds m_SpawnAreaBounds;


    // public GameObject goal;
    public GameObject shortBlock;
    public GameObject wall;
    Rigidbody m_ShortBlockRb;
    Rigidbody m_AgentRb;
    Material m_GroundMaterial;
    Renderer m_GroundRenderer;
    WallJumpAcademy m_Academy;
    RayPerception m_RayPer;

    // public float jumpingTime;
    public float jumpTimer; //timer used to control jump timing & prevent rapid doublejumps
    // // This is a downward force applied when falling to make jumps look
    // // less floaty
    // public float fallingForce;
    // // Use to check the coliding objects
    // public Collider[] hitGroundColliders = new Collider[3];
    // Vector3 m_JumpTargetPos;
    // Vector3 m_JumpStartingPos;

    string[] m_DetectableObjects;
    // public bool grounded;
    // public bool largeGround;
    public float m_currentVelMag;

    //Groundcheck
    private AgentCubeGroundCheck m_groundCheck;
    public AgentCubeMovement m_agentMovement;
    int fuTimer;
    public float smoothTime = 0.3F;
    private Vector3 velocity = Vector3.zero;

    public override void InitializeAgent()
    {
        m_groundCheck = GetComponent<AgentCubeGroundCheck>();
        m_Academy = FindObjectOfType<WallJumpAcademy>();
        m_agentMovement = FindObjectOfType<AgentCubeMovement>();
        m_RayPer = GetComponent<RayPerception>();
        m_Configuration = Random.Range(0, 5);
        m_DetectableObjects = new[] { "wall", "goal", "block" };

        m_AgentRb = GetComponent<Rigidbody>();
        m_ShortBlockRb = shortBlock.GetComponent<Rigidbody>();
        m_SpawnAreaBounds = spawnArea.GetComponent<Collider>().bounds;
        m_GroundRenderer = ground.GetComponent<Renderer>();
        m_GroundMaterial = m_GroundRenderer.material;
        

        spawnArea.SetActive(false);
    }

    // Begin the jump sequence
    // public void Jump()
    // {
    //     jumpingTime = 0.2f;
    //     m_JumpStartingPos = m_AgentRb.position;
    // }

    // /// <summary>
    // /// Does the ground check.
    // /// </summary>
    // /// <returns><c>true</c>, if the agent is on the ground,
    // /// <c>false</c> otherwise.</returns>
    // /// <param name="smallCheck"></param>
    // public bool DoGroundCheck(bool smallCheck)
    // {
    //     if (!smallCheck)
    //     {
    //         hitGroundColliders = new Collider[3];
    //         var o = gameObject;
    //         Physics.OverlapBoxNonAlloc(
    //             o.transform.position + new Vector3(0, -0.05f, 0),
    //             new Vector3(0.95f / 2f, 0.5f, 0.95f / 2f),
    //             hitGroundColliders,
    //             o.transform.rotation);
    //         var grounded = false;
    //         foreach (var col in hitGroundColliders)
    //         {
    //             // if (col != null && col.transform != transform &&
    //             //     (col.CompareTag("walkableSurface") ||
    //             //      col.CompareTag("block") ||
    //             //      col.CompareTag("wall")))
    //             // {
    //             if (col != null && col.transform != transform &&
    //                 (col.CompareTag("walkableSurface") ||
    //                  col.CompareTag("block")))
    //             {
    //                 grounded = true; //then we're grounded
    //                 break;
    //             }
    //         }
    //         return grounded;
    //     }
    //     else
    //     {
    //         RaycastHit hit;
    //         // Physics.SphereCast(m_AgentRb.position + new Vector3(0, .5f, 0), .55f, -Vector3.up, out hit,
    //         //     1f);
    //         Physics.Raycast(transform.position + new Vector3(0, -0.05f, 0), -Vector3.up, out hit,
    //             .51f);

    //         // if (hit.collider != null &&
    //         //     (hit.collider.CompareTag("walkableSurface") ||
    //         //      hit.collider.CompareTag("block") ||
    //         //      hit.collider.CompareTag("wall"))
    //         //     && hit.normal.y > 0.95f)
    //         // {
    //         if (hit.collider != null &&
    //             (hit.collider.CompareTag("walkableSurface") ||
    //              hit.collider.CompareTag("block"))
    //             && hit.normal.y > 0.95f)
    //         {
    //             return true;
    //         }

    //         return false;
    //     }
    // }

    // /// <summary>
    // /// Moves  a rigidbody towards a position smoothly.
    // /// </summary>
    // /// <param name="targetPos">Target position.</param>
    // /// <param name="rb">The rigidbody to be moved.</param>
    // /// <param name="targetVel">The velocity to target during the
    // ///  motion.</param>
    // /// <param name="maxVel">The maximum velocity posible.</param>
    // void MoveTowards(
    //     Vector3 targetPos, Rigidbody rb, float targetVel, float maxVel)
    // {
    //     var moveToPos = targetPos - rb.worldCenterOfMass;
    //     var velocityTarget = Time.fixedDeltaTime * targetVel * moveToPos;
    //     if (float.IsNaN(velocityTarget.x) == false)
    //     {
    //         rb.velocity = Vector3.MoveTowards(
    //             rb.velocity, velocityTarget, maxVel);
    //     }
    // }

    public override void CollectObservations()
    {
        // grounded = DoGroundCheck(true);
        // largeGround = DoGroundCheck(false);
        // print($"Observation: {m_Academy.GetStepCount()} {fuTimer}");
        // print($"Observation: {GetStepCount()} {fuTimer}");

        var rayDistance = 20f;
        float[] rayAngles = { 0f, 45f, 90f, 135f, 180f, 110f, 70f };
        AddVectorObs(m_RayPer.Perceive(
            rayDistance, rayAngles, m_DetectableObjects, 0f, 0f));
        AddVectorObs(m_RayPer.Perceive(
            rayDistance, rayAngles, m_DetectableObjects, 2.5f, 2.5f));
        var agentPos = m_AgentRb.position - ground.transform.position;

        AddVectorObs(agentPos / 20f);
        // AddVectorObs(grounded? 1 : 0);
        AddVectorObs(m_groundCheck.isGrounded);
        AddVectorObs(m_AgentRb.velocity/m_agentMovement.agentRunSpeed);
        AddVectorObs(m_AgentRb.angularVelocity/m_AgentRb.maxAngularVelocity);
        AddVectorObs(m_AgentRb.transform.forward);
    }

    /// <summary>
    /// Gets a random spawn position in the spawningArea.
    /// </summary>
    /// <returns>The random spawn position.</returns>
    public Vector3 GetRandomSpawnPos()
    {
        var randomPosX = Random.Range(-m_SpawnAreaBounds.extents.x,
            m_SpawnAreaBounds.extents.x);
        var randomPosZ = Random.Range(-m_SpawnAreaBounds.extents.z,
            m_SpawnAreaBounds.extents.z);

        var randomSpawnPos = spawnArea.transform.position +
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
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time);
        m_GroundRenderer.material = m_GroundMaterial;
    }

    public void MoveAgent(float[] act)
    {
        AddReward(-0.0005f);
        // var smallGrounded = DoGroundCheck(true);
        // var smallGrounded = grounded;
        // var largeGrounded = largeGround;
        // largeGround = largeGrounded;

        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;
        var dirToGoForwardAction = (int)act[0];
        var rotateDirAction = (int)act[1];
//        var dirToGoSideAction = (int)act[2];
        var jumpAction = (int)act[2];
        // print($"Academy: {m_Academy.GetStepCount()}");
        // print($"Agent: {GetStepCount()}");
        // print($"Action: {GetStepCount()} {fuTimer}");
        // if (smallGrounded)
        // {
        //     if (dirToGoForwardAction == 1)
        //         dirToGo = 1f * transform.forward;
        //     else if (dirToGoForwardAction == 2)
        //         dirToGo = -1f * transform.forward;
        //     if (dirToGoSideAction == 1)
        //         dirToGo = -0.6f * transform.right;
        //     else if (dirToGoSideAction == 2)
        //         dirToGo = 0.6f * transform.right;
        // }
        // // if (smallGrounded)
        // // {
            // if (dirToGoForwardAction == 1)
            //     dirToGo = (largeGrounded ? 1f : 0.1f) * 1f * transform.forward;
            // else if (dirToGoForwardAction == 2)
            //     dirToGo = (largeGrounded ? 1f : 0.1f) * -1f * transform.forward;
            // if (dirToGoSideAction == 1)
            //     dirToGo = (largeGrounded ? 1f : 0.1f) * -0.6f * transform.right;
            // else if (dirToGoSideAction == 2)
            //     dirToGo = (largeGrounded ? 1f : 0.1f) * 0.6f * transform.right;
            if (dirToGoForwardAction == 1)
                dirToGo += transform.forward;
            else if (dirToGoForwardAction == 2)
                dirToGo += -transform.forward;
//            if (dirToGoSideAction == 1)
//                dirToGo += -transform.right;
//            else if (dirToGoSideAction == 2)
//                dirToGo += transform.right;
        // // }
        if (rotateDirAction == 1)
            rotateDir = -transform.up ;
        else if (rotateDirAction == 2)
            rotateDir = transform.up;
        if (jumpAction == 1)
        {
            // if (largeGrounded)
//            if (m_groundCheck.isGrounded && jumpTimer > .15f)
            if (m_groundCheck.isGrounded)
            {
                // Jump();
                // print($"JUMP {m_groundCheck.isGrounded}");
                // m_AgentRb.velocity = Vector3.zero;
                
//                jumpTimer = 0f;
                m_agentMovement.Jump(m_AgentRb);
//                Vector3 velToUse = m_AgentRb.velocity;
//                velToUse.y = m_Academy.agentJumpVelocity;
//                m_AgentRb.velocity = velToUse;


//                m_AgentRb.AddForce(Vector3.up * m_Academy.agentJumpVelocity,
//                    ForceMode.VelocityChange);
//                m_AgentRb.AddForce(Vector3.up * m_Academy.agentJumpVelocity,
//                    ForceMode.Impulse);
                // AddReward(-0.01f); //don't constantly jump
            }
        }

        // transform.Rotate(rotateDir, Time.fixedDeltaTime * 300f);
//        transform.Rotate(rotateDir, m_Academy.agentRotationSpeed);
        if (rotateDir != Vector3.zero)
        {
            m_agentMovement.RotateBody(m_AgentRb, rotateDir);
//            m_AgentRb.MoveRotation(m_AgentRb.rotation * Quaternion.AngleAxis(m_Academy.agentRotationSpeed, rotateDir));
        }

        
        //Running Logic
//        float runSpeed = Mathf.Clamp(m_agentMovement.agentRunSpeed - m_currentVelMag, 0, m_agentMovement.agentRunSpeed);
//        float runSpeed = Mathf.Clamp(m_agentMovement.agentRunMaxVelMagnitude - m_currentVelMag, 0, m_agentMovement.agentRunMaxVelMagnitude);
        if (!m_groundCheck.isGrounded)
        {
            m_agentMovement.RunInAir(m_AgentRb, dirToGo.normalized);
        }
        else
        {
            m_agentMovement.RunOnGround(m_AgentRb, dirToGo.normalized);
        }

        
//        m_AgentRb.AddForce(dirToGo.normalized * runSpeed,
//            ForceMode.VelocityChange); 
//        m_AgentRb.AddForce(dirToGo.normalized * runSpeed,
//            ForceMode.Impulse); 
        
//        if (m_groundCheck.isGrounded && dirToGo == Vector3.zero && jumpTimer > .15f)
        if (m_groundCheck.isGrounded && dirToGo == Vector3.zero)
        {
            m_agentMovement.AddIdleDrag(m_AgentRb);
//            m_AgentRb.velocity *= m_Academy.agentIdleDragVelCoeff;
//            print("damping");
        }
//        //Running Logic
//        Vector3 smoothedVel = Vector3.SmoothDamp(m_AgentRb.velocity, dirToGo * runSpeed, ref velocity, smoothTime);
//        m_AgentRb.velocity = smoothedVel;
        
        
        
        
        
        
        // if(m_currentVelMag < m_Academy.agentMaxVel)
        // {
        //     m_AgentRb.AddForce(dirToGo * m_Academy.agentRunSpeed,
        //         ForceMode.VelocityChange);
        // }

        // if (jumpingTime > 0f)
        // {
        //     m_JumpTargetPos =
        //         new Vector3(m_AgentRb.position.x,
        //             m_JumpStartingPos.y + m_Academy.agentJumpHeight,
        //             m_AgentRb.position.z) + dirToGo;
        //     MoveTowards(m_JumpTargetPos, m_AgentRb, m_Academy.agentJumpVelocity,
        //         m_Academy.agentJumpVelocityMaxChange);
        // }

//        if (!m_groundCheck.isGrounded && jumpTimer > .15f)
        if (!m_groundCheck.isGrounded)
        {
            m_agentMovement.AddFallingForce(m_AgentRb);
//            m_AgentRb.AddForce(
//                // Vector3.down * fallingForce, ForceMode.Acceleration);
//                Vector3.down * m_Academy.agentFallingSpeed, ForceMode.Acceleration);
        }
        // if (!(jumpingTime > 0f) && !m_groundCheck.isGrounded)
        // {
        //     m_AgentRb.AddForce(
        //         // Vector3.down * fallingForce, ForceMode.Acceleration);
        //         Vector3.down * m_Academy.agentFallingSpeed, ForceMode.Acceleration);
        // }
        // jumpingTime -= Time.fixedDeltaTime;
//        jumpTimer += Time.fixedDeltaTime;
    }
    // public void MoveAgent(float[] act)
    // {
    //     AddReward(-0.0005f);
    //     var smallGrounded = DoGroundCheck(true);
    //     grounded = smallGrounded;
    //     var largeGrounded = DoGroundCheck(false);

    //     var dirToGo = Vector3.zero;
    //     var rotateDir = Vector3.zero;
    //     var dirToGoForwardAction = (int)act[0];
    //     var rotateDirAction = (int)act[1];
    //     var dirToGoSideAction = (int)act[2];
    //     var jumpAction = (int)act[3];

    //     if (dirToGoForwardAction == 1)
    //         dirToGo = (largeGrounded ? 1f : 0.5f) * 1f * transform.forward;
    //     else if (dirToGoForwardAction == 2)
    //         dirToGo = (largeGrounded ? 1f : 0.5f) * -1f * transform.forward;
    //     if (rotateDirAction == 1)
    //         rotateDir = transform.up * -1f;
    //     else if (rotateDirAction == 2)
    //         rotateDir = transform.up * 1f;
    //     if (dirToGoSideAction == 1)
    //         dirToGo = (largeGrounded ? 1f : 0.5f) * -0.6f * transform.right;
    //     else if (dirToGoSideAction == 2)
    //         dirToGo = (largeGrounded ? 1f : 0.5f) * 0.6f * transform.right;
    //     if (jumpAction == 1)
    //         if ((jumpingTime <= 0f) && smallGrounded)
    //         {
    //             Jump();
    //             AddReward(-0.01f); //don't constantly jump
    //         }

    //     transform.Rotate(rotateDir, Time.fixedDeltaTime * 300f);
    //     m_AgentRb.AddForce(dirToGo * m_Academy.agentRunSpeed,
    //         ForceMode.VelocityChange);

    //     if (jumpingTime > 0f)
    //     {
    //         m_JumpTargetPos =
    //             new Vector3(m_AgentRb.position.x,
    //                 m_JumpStartingPos.y + m_Academy.agentJumpHeight,
    //                 m_AgentRb.position.z) + dirToGo;
    //         MoveTowards(m_JumpTargetPos, m_AgentRb, m_Academy.agentJumpVelocity,
    //             m_Academy.agentJumpVelocityMaxChange);
    //     }

    //     if (!(jumpingTime > 0f) && !largeGrounded)
    //     {
    //         m_AgentRb.AddForce(
    //             Vector3.down * fallingForce, ForceMode.Acceleration);
    //     }
    //     jumpingTime -= Time.fixedDeltaTime;
    // }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        MoveAgent(vectorAction);
        // if ((!Physics.Raycast(m_AgentRb.position, Vector3.down, 20))
        //     || (!Physics.Raycast(m_ShortBlockRb.position, Vector3.down, 20)))
        if(m_AgentRb.position.y < -1 || m_ShortBlockRb.position.y < -1)
        {
            SetReward(-1f);
            ResetBlock(m_ShortBlockRb);
            StartCoroutine(
                GoalScoredSwapGroundMaterial(m_Academy.failMaterial, .5f));
            Done();
        }
    }

    // Detect when the agent hits the goal
    // void OnTriggerEnter(Collider col)
    void OnTriggerStay(Collider col)
    {
        // if (col.gameObject.CompareTag("goal") && DoGroundCheck(true))
        if (col.gameObject.CompareTag("goal") && m_groundCheck.isGrounded)
        {
            SetReward(1f);
            ResetBlock(m_ShortBlockRb);
            StartCoroutine(
                GoalScoredSwapGroundMaterial(m_Academy.goalScoredMaterial, .5f));
            Done();
        }
    }

    //Reset the orange block position
    void ResetBlock(Rigidbody blockRb)
    {
        blockRb.transform.position = GetRandomSpawnPos();
        blockRb.velocity = Vector3.zero;
        blockRb.angularVelocity = Vector3.zero;
    }

    public override void AgentReset()
    {
        ResetBlock(m_ShortBlockRb);
        transform.localPosition = new Vector3(
            18 * (Random.value - 0.5f), 1, -12);
        m_Configuration = Random.Range(0, 5);
        // m_AgentRb.velocity = default(Vector3);
        m_AgentRb.velocity = Vector3.zero;
    }

    private void FixedUpdate()
    {
        m_currentVelMag = m_AgentRb.velocity.magnitude;
//        fuTimer++;
        if (m_Configuration != -1)
        {
            ConfigureAgent(m_Configuration);
            m_Configuration = -1;
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
        var localScale = wall.transform.localScale;
        if (config == 0)
        {
            localScale = new Vector3(
                localScale.x,
                m_Academy.resetParameters["no_wall_height"],
                localScale.z);
            wall.transform.localScale = localScale;
            GiveBrain(noWallBrain);
        }
        else if (config == 1)
        {
            localScale = new Vector3(
                localScale.x,
                m_Academy.resetParameters["small_wall_height"],
                localScale.z);
            wall.transform.localScale = localScale;
            GiveBrain(smallWallBrain);
        }
        else
        {
            var height =
                m_Academy.resetParameters["big_wall_min_height"] +
                Random.value * (m_Academy.resetParameters["big_wall_max_height"] -
                    m_Academy.resetParameters["big_wall_min_height"]);
            localScale = new Vector3(
                localScale.x,
                height,
                localScale.z);
            wall.transform.localScale = localScale;
            GiveBrain(bigWallBrain);
        }
    }
}
