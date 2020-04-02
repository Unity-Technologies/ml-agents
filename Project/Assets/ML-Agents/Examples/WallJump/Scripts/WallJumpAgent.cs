//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using MLAgents;
using Barracuda;
using MLAgents.Sensors;
using MLAgents.SideChannels;

public class WallJumpAgent : Agent
{
    // Depending on this value, the wall will have different height
    int m_Configuration;
    // Brain to use when no wall is present
    public NNModel noWallBrain;
    // Brain to use when a jumpable wall is present
    public NNModel smallWallBrain;
    // Brain to use when a wall requiring a block to jump over is present
    public NNModel bigWallBrain;

    public GameObject ground;
    public GameObject spawnArea;
    Bounds m_SpawnAreaBounds;


    public GameObject goal;
    public GameObject shortBlock;
    public GameObject wall;
    Rigidbody m_ShortBlockRb;
    Rigidbody m_AgentRb;
    Material m_GroundMaterial;
    Renderer m_GroundRenderer;
    WallJumpSettings m_WallJumpSettings;

    public float jumpingTime;
    public float jumpTime;
    // This is a downward force applied when falling to make jumps look
    // less floaty
    public float fallingForce;
    // Use to check the coliding objects
    public Collider[] hitGroundColliders = new Collider[3];
    Vector3 m_JumpTargetPos;
    Vector3 m_JumpStartingPos;

    FloatPropertiesChannel m_FloatProperties;

    public override void Initialize()
    {
        m_WallJumpSettings = FindObjectOfType<WallJumpSettings>();
        m_Configuration = Random.Range(0, 5);

        m_AgentRb = GetComponent<Rigidbody>();
        m_ShortBlockRb = shortBlock.GetComponent<Rigidbody>();
        m_SpawnAreaBounds = spawnArea.GetComponent<Collider>().bounds;
        m_GroundRenderer = ground.GetComponent<Renderer>();
        m_GroundMaterial = m_GroundRenderer.material;

        spawnArea.SetActive(false);

        m_FloatProperties = SideChannelUtils.GetSideChannel<FloatPropertiesChannel>();
    }

    // Begin the jump sequence
    public void Jump()
    {
        jumpingTime = 0.2f;
        m_JumpStartingPos = m_AgentRb.position;
    }

    /// <summary>
    /// Does the ground check.
    /// </summary>
    /// <returns><c>true</c>, if the agent is on the ground,
    /// <c>false</c> otherwise.</returns>
    /// <param name="smallCheck"></param>
    public bool DoGroundCheck(bool smallCheck)
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
            var grounded = false;
            foreach (var col in hitGroundColliders)
            {
                if (col != null && col.transform != transform &&
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
        else
        {
            RaycastHit hit;
            Physics.Raycast(transform.position + new Vector3(0, -0.05f, 0), -Vector3.up, out hit,
                1f);

            if (hit.collider != null &&
                (hit.collider.CompareTag("walkableSurface") ||
                 hit.collider.CompareTag("block") ||
                 hit.collider.CompareTag("wall"))
                && hit.normal.y > 0.95f)
            {
                return true;
            }

            return false;
        }
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
        var moveToPos = targetPos - rb.worldCenterOfMass;
        var velocityTarget = Time.fixedDeltaTime * targetVel * moveToPos;
        if (float.IsNaN(velocityTarget.x) == false)
        {
            rb.velocity = Vector3.MoveTowards(
                rb.velocity, velocityTarget, maxVel);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        var agentPos = m_AgentRb.position - ground.transform.position;

        sensor.AddObservation(agentPos / 20f);
        sensor.AddObservation(DoGroundCheck(true) ? 1 : 0);
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
    /// Changes the color of the ground for a moment.
    /// </summary>
    /// <returns>The Enumerator to be used in a Coroutine.</returns>
    /// <param name="mat">The material to be swapped.</param>
    /// <param name="time">The time the material will remain.</param>
    IEnumerator GoalScoredSwapGroundMaterial(Material mat, float time)
    {
        m_GroundRenderer.material = mat;
        yield return new WaitForSeconds(time); //wait for 2 sec
        m_GroundRenderer.material = m_GroundMaterial;
    }

    public void MoveAgent(float[] act)
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
        m_AgentRb.AddForce(dirToGo * m_WallJumpSettings.agentRunSpeed,
            ForceMode.VelocityChange);

        if (jumpingTime > 0f)
        {
            m_JumpTargetPos =
                new Vector3(m_AgentRb.position.x,
                    m_JumpStartingPos.y + m_WallJumpSettings.agentJumpHeight,
                    m_AgentRb.position.z) + dirToGo;
            MoveTowards(m_JumpTargetPos, m_AgentRb, m_WallJumpSettings.agentJumpVelocity,
                m_WallJumpSettings.agentJumpVelocityMaxChange);
        }

        if (!(jumpingTime > 0f) && !largeGrounded)
        {
            m_AgentRb.AddForce(
                Vector3.down * fallingForce, ForceMode.Acceleration);
        }
        jumpingTime -= Time.fixedDeltaTime;
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        MoveAgent(vectorAction);
        if ((!Physics.Raycast(m_AgentRb.position, Vector3.down, 20))
            || (!Physics.Raycast(m_ShortBlockRb.position, Vector3.down, 20)))
        {
            SetReward(-1f);
            EndEpisode();
            ResetBlock(m_ShortBlockRb);
            StartCoroutine(
                GoalScoredSwapGroundMaterial(m_WallJumpSettings.failMaterial, .5f));
        }
    }

    public override float[] Heuristic()
    {
        var action = new float[4];
        if (Input.GetKey(KeyCode.D))
        {
            action[1] = 2f;
        }
        if (Input.GetKey(KeyCode.W))
        {
            action[0] = 1f;
        }
        if (Input.GetKey(KeyCode.A))
        {
            action[1] = 1f;
        }
        if (Input.GetKey(KeyCode.S))
        {
            action[0] = 2f;
        }
        action[3] = Input.GetKey(KeyCode.Space) ? 1.0f : 0.0f;
        return action;
    }

    // Detect when the agent hits the goal
    void OnTriggerStay(Collider col)
    {
        if (col.gameObject.CompareTag("goal") && DoGroundCheck(true))
        {
            SetReward(1f);
            EndEpisode();
            StartCoroutine(
                GoalScoredSwapGroundMaterial(m_WallJumpSettings.goalScoredMaterial, 2));
        }
    }

    //Reset the orange block position
    void ResetBlock(Rigidbody blockRb)
    {
        blockRb.transform.position = GetRandomSpawnPos();
        blockRb.velocity = Vector3.zero;
        blockRb.angularVelocity = Vector3.zero;
    }

    public override void OnEpisodeBegin()
    {
        ResetBlock(m_ShortBlockRb);
        transform.localPosition = new Vector3(
            18 * (Random.value - 0.5f), 1, -12);
        m_Configuration = Random.Range(0, 5);
        m_AgentRb.velocity = default(Vector3);
    }

    void FixedUpdate()
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
    /// Other : Tall wall and BigWallBrain.
    /// </param>
    void ConfigureAgent(int config)
    {
        var localScale = wall.transform.localScale;
        if (config == 0)
        {
            localScale = new Vector3(
                localScale.x,
                m_FloatProperties.GetPropertyWithDefault("no_wall_height", 0),
                localScale.z);
            wall.transform.localScale = localScale;
            SetModel("SmallWallJump", noWallBrain);
        }
        else if (config == 1)
        {
            localScale = new Vector3(
                localScale.x,
                m_FloatProperties.GetPropertyWithDefault("small_wall_height", 4),
                localScale.z);
            wall.transform.localScale = localScale;
            SetModel("SmallWallJump", smallWallBrain);
        }
        else
        {
            var min = m_FloatProperties.GetPropertyWithDefault("big_wall_min_height", 8);
            var max = m_FloatProperties.GetPropertyWithDefault("big_wall_max_height", 8);
            var height = min + Random.value * (max - min);
            localScale = new Vector3(
                localScale.x,
                height,
                localScale.z);
            wall.transform.localScale = localScale;
            SetModel("BigWallJump", bigWallBrain);
        }
    }
}
