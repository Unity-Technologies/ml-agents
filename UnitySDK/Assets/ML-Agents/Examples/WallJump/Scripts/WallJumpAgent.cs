//Put this script on your blue cube.

using System.Collections;
using System.Linq;
using UnityEngine;
using MLAgents;

public class WallJumpAgent : Agent
{
    // Depending on this value, the wall will have different height
    int m_Configuration;
    // Brain to use when no wall is present
    public Brain noWallBrain;
    // Brain to use when a jump-able wall is present
    public Brain smallWallBrain;
    // Brain to use when a wall requiring a block to jump over is present
    public Brain bigWallBrain;

    public GameObject ground;
    public GameObject spawnArea;
    Bounds m_SpawnAreaBounds;

    public GameObject shortBlock;
    public GameObject wall;
    Rigidbody m_ShortBlockRb;
    Rigidbody m_AgentRb;
    Material m_GroundMaterial;
    Renderer m_GroundRenderer;
    WallJumpAcademy m_Academy;
    RayPerception m_RayPer;

    public float jumpingTime;

    // This is a downward force applied when falling to make jumps look
    // less floaty
    public float fallingForce;
    // Use to check the colliding objects
    public Collider[] hitGroundColliders = new Collider[3];
    Vector3 m_JumpTargetPos;
    Vector3 m_JumpStartingPos;

    string[] m_DetectableObjects;

    protected override void InitializeAgent()
    {
        m_Academy = FindObjectOfType<WallJumpAcademy>();
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
    void Jump()
    {
        jumpingTime = 0.2f;
        m_JumpStartingPos = m_AgentRb.position;
    }

    /// <summary>
    /// Does the ground check.
    /// </summary>
    /// <returns><c>true</c>, if the agent is on the ground,
    /// <c>false</c> otherwise.</returns>
    bool DoGroundCheck(bool smallCheck)
    {
        if (!smallCheck)
        {
            hitGroundColliders = new Collider[3];
            var o = gameObject;
            Physics.OverlapBoxNonAlloc(
                o.transform.position + new Vector3(0, -0.05f, 0),
                new Vector3(0.95f / 2f, 0.5f, 0.95f / 2f),
                hitGroundColliders,
                o.transform.rotation);
            return hitGroundColliders.Any(col =>
                col != null &&
                col.transform != transform &&
                // ReSharper disable once Unity.UnknownTag (the tag is in the project)
                (col.CompareTag("walkableSurface") || col.CompareTag("block") || col.CompareTag("wall")));
        }

        RaycastHit hit;
        Physics.Raycast(transform.position + new Vector3(0, -0.05f, 0), -Vector3.up, out hit,
            1f);

        return hit.collider != null &&
            // ReSharper disable once Unity.UnknownTag (the tag is in the project)
            (hit.collider.CompareTag("walkableSurface") ||
                hit.collider.CompareTag("block") ||
                hit.collider.CompareTag("wall"))
            && hit.normal.y > 0.95f;
    }

    /// <summary>
    /// Moves  a rigid body towards a position smoothly.
    /// </summary>
    /// <param name="targetPos">Target position.</param>
    /// <param name="rb">The rigid body to be moved.</param>
    /// <param name="targetVel">The velocity to target during the
    ///  motion.</param>
    /// <param name="maxVel">The maximum velocity possible.</param>
    static void MoveTowards(
        Vector3 targetPos, Rigidbody rb, float targetVel, float maxVel)
    {
        var moveToPos = targetPos - rb.worldCenterOfMass;
        var velocityTarget = Time.fixedDeltaTime * targetVel * moveToPos;
        if (float.IsNaN(velocityTarget.x) == false)
        {
            rb.velocity = Vector3.MoveTowards(
                rb.velocity, velocityTarget, maxVel);
        }
    }

    protected override void CollectObservations()
    {
        const float rayDistance = 20f;
        float[] rayAngles = { 0f, 45f, 90f, 135f, 180f, 110f, 70f };
        AddVectorObs(m_RayPer.Perceive(
            rayDistance, rayAngles, m_DetectableObjects, 0f, 0f));
        AddVectorObs(m_RayPer.Perceive(
            rayDistance, rayAngles, m_DetectableObjects, 2.5f, 2.5f));
        Vector3 agentPos = m_AgentRb.position - ground.transform.position;

        AddVectorObs(agentPos / 20f);
        AddVectorObs(DoGroundCheck(true) ? 1 : 0);
    }

    /// <summary>
    /// Gets a random spawn position in the spawningArea.
    /// </summary>
    /// <returns>The random spawn position.</returns>
    Vector3 GetRandomSpawnPos()
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
    /// Changes the color of the ground for a moment
    /// </summary>
    /// <returns>The Enumerator to be used in a Coroutine</returns>
    /// <param name="mat">The material to be swapped.</param>
    /// <param name="time">The time the material will remain.</param>
    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time); //wait for 2 sec
        // ReSharper disable once Unity.InefficientPropertyAccess
        m_GroundRenderer.material = m_GroundMaterial;
    }

    void MoveAgent(float[] act)
    {
        AddReward(-0.0005f);
        var smallGrounded = DoGroundCheck(true);
        var largeGrounded = DoGroundCheck(false);

        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;
        var dirToGoForwardAction = (int)act[0];
        var rotateDirAction = (int)act[1];
        var dirToGoSideAction = (int)act[2];
        var jumpAction = (int)act[3];

        if (dirToGoForwardAction == 1)
            dirToGo = (largeGrounded ? 1f : 0.5f) * 1f * transform.forward;
        else if (dirToGoForwardAction == 2)
            dirToGo = (largeGrounded ? 1f : 0.5f) * -1f * transform.forward;
        if (rotateDirAction == 1)
            rotateDir = transform.up * -1f;
        else if (rotateDirAction == 2)
            rotateDir = transform.up * 1f;
        if (dirToGoSideAction == 1)
            dirToGo = (largeGrounded ? 1f : 0.5f) * -0.6f * transform.right;
        else if (dirToGoSideAction == 2)
            dirToGo = (largeGrounded ? 1f : 0.5f) * 0.6f * transform.right;
        if (jumpAction == 1)
            if ((jumpingTime <= 0f) && smallGrounded)
            {
                Jump();
            }

        transform.Rotate(rotateDir, Time.fixedDeltaTime * 300f);
        m_AgentRb.AddForce(dirToGo * m_Academy.agentRunSpeed,
            ForceMode.VelocityChange);

        if (jumpingTime > 0f)
        {
            var position = m_AgentRb.position;
            m_JumpTargetPos =
                new Vector3(position.x,
                    m_JumpStartingPos.y + m_Academy.agentJumpHeight,
                    position.z) + dirToGo;
            MoveTowards(m_JumpTargetPos, m_AgentRb, m_Academy.agentJumpVelocity,
                m_Academy.agentJumpVelocityMaxChange);
        }

        if (!(jumpingTime > 0f) && !largeGrounded)
        {
            m_AgentRb.AddForce(
                Vector3.down * fallingForce, ForceMode.Acceleration);
        }
        jumpingTime -= Time.fixedDeltaTime;
    }

    public override void AgentAction(float[] vectorAction, string textAction)
    {
        MoveAgent(vectorAction);
        if ((!Physics.Raycast(m_AgentRb.position, Vector3.down, 20))
            || (!Physics.Raycast(m_ShortBlockRb.position, Vector3.down, 20)))
        {
            Done();
            SetReward(-1f);
            ResetBlock(m_ShortBlockRb);
            StartCoroutine(
                GoalScoredSwapGroundMaterial(m_Academy.failMaterial, .5f));
        }
    }

    // Detect when the agent hits the goal
    void OnTriggerStay(Collider col)
    {
        if (col.gameObject.CompareTag("goal") && DoGroundCheck(true))
        {
            SetReward(1f);
            Done();
            StartCoroutine(
                GoalScoredSwapGroundMaterial(m_Academy.goalScoredMaterial, 2));
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
        m_AgentRb.velocity = default(Vector3);
    }

    private void FixedUpdate()
    {
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
            var scale = wall.transform.localScale;
            scale = new Vector3(
                scale.x,
                height,
                scale.z);
            wall.transform.localScale = scale;
            GiveBrain(bigWallBrain);
        }
    }
}
