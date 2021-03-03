//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.Barracuda;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgentsExamples;

public class WJPBAgent : Agent
{
    // Depending on this value, the agent will have different goals
    int m_Goal;

    public GameObject ground;
    public GameObject spawnArea;
    Bounds m_SpawnAreaBounds;
    VectorSensorComponent goalSensor;

    public GameObject wallJumpGoal;
    public GameObject pushBlockGoal;
    public GameObject shortBlock;
    public GameObject wall;
    public int goals;
    Rigidbody m_ShortBlockRb;
    Rigidbody m_AgentRb;
    public Material goalMaterial;
    public Material nonGoalMaterial;
    Renderer m_PushRenderer;
    Renderer m_WallRenderer;
    WJPBSettings m_WJPBSettings;

    public float jumpingTime;
    public float jumpTime;
    // This is a downward force applied when falling to make jumps look
    // less floaty
    public float fallingForce;
    // Use to check the coliding objects
    public Collider[] hitGroundColliders = new Collider[3];
    Vector3 m_JumpTargetPos;
    Vector3 m_JumpStartingPos;

    [HideInInspector]
    public WJPBGoalDetect goalDetect;
    float[] m_GoalOneHot;


    EnvironmentParameters m_ResetParams;

    public override void Initialize()
    {
        m_WJPBSettings = FindObjectOfType<WJPBSettings>();
        // One-hot encoding of the goal
        m_WallRenderer = wallJumpGoal.GetComponent<Renderer>();
        m_PushRenderer = pushBlockGoal.GetComponent<Renderer>();

        m_Goal = Random.Range(0, goals);
        //m_Goal = 0;
        m_GoalOneHot = new float[goals];
        if (m_Goal == 0)
        {
            m_WallRenderer.material = goalMaterial;
            m_PushRenderer.material = nonGoalMaterial;
        }
        else
        {
            m_WallRenderer.material = nonGoalMaterial;
            m_PushRenderer.material = goalMaterial;
        }

        System.Array.Clear(m_GoalOneHot, 0, m_GoalOneHot.Length);
        m_GoalOneHot[m_Goal] = 1;

        goalDetect = shortBlock.GetComponent<WJPBGoalDetect>();
        goalDetect.agent = this;


        m_AgentRb = GetComponent<Rigidbody>();
        m_ShortBlockRb = shortBlock.GetComponent<Rigidbody>();
        m_SpawnAreaBounds = spawnArea.GetComponent<Collider>().bounds;

        spawnArea.SetActive(false);

        m_ResetParams = Academy.Instance.EnvironmentParameters;

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
        goalSensor = this.GetComponent<VectorSensorComponent>();
        goalSensor.GetSensor().AddObservation(m_GoalOneHot);
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
            new Vector3(randomPosX, 0.45f, randomPosZ - 2f);
        return randomSpawnPos;
    }

    public void MoveAgent(ActionSegment<int> act)
    {
        AddReward(-0.0002f);
        var smallGrounded = DoGroundCheck(true);
        var largeGrounded = DoGroundCheck(false);

        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;
        var dirToGoForwardAction = act[0];
        var rotateDirAction = act[1];
        var dirToGoSideAction = act[2];
        var jumpAction = act[3];

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
        m_AgentRb.AddForce(dirToGo * m_WJPBSettings.agentRunSpeed,
            ForceMode.VelocityChange);

        if (jumpingTime > 0f)
        {
            m_JumpTargetPos =
                new Vector3(m_AgentRb.position.x,
                    m_JumpStartingPos.y + m_WJPBSettings.agentJumpHeight,
                    m_AgentRb.position.z) + dirToGo;
            MoveTowards(m_JumpTargetPos, m_AgentRb, m_WJPBSettings.agentJumpVelocity,
                m_WJPBSettings.agentJumpVelocityMaxChange);
        }

        if (!(jumpingTime > 0f) && !largeGrounded)
        {
            m_AgentRb.AddForce(
                Vector3.down * fallingForce, ForceMode.Acceleration);
        }
        jumpingTime -= Time.fixedDeltaTime;
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        MoveAgent(actionBuffers.DiscreteActions);
        if ((!Physics.Raycast(m_AgentRb.position, Vector3.down, 20))
            || (!Physics.Raycast(m_ShortBlockRb.position, Vector3.down, 20)))
        {
            SetReward(-1f);
            EndEpisode();
            ResetBlock(m_ShortBlockRb);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut.Clear();
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[1] = 2;
        }
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[1] = 1;
        }
        if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
        discreteActionsOut[3] = Input.GetKey(KeyCode.Space) ? 1 : 0;
    }

    // Detect when the agent hits the goal
    void OnTriggerStay(Collider col)
    {
        if (col.gameObject.CompareTag("walljumpgoal") && DoGroundCheck(true))
        {
            // ending the episode so the agent doesnt get stuck over the wall
            if (m_Goal == 0)
            {
                SetReward(1f);
                EndEpisode();
            }
            else
            {
                SetReward(-1f);
                EndEpisode();
            }
        }
    }

    public void ScoredAGoal()
    {
        if (m_Goal == 1)
        {
            SetReward(1f);
            EndEpisode();
        }
        else
        {
            SetReward(-1f);
            EndEpisode();
        }
    }


    //Reset the orange block position
    void ResetBlock(Rigidbody blockRb)
    {
        blockRb.transform.position = GetRandomSpawnPos();
        blockRb.velocity = Vector3.zero;
        blockRb.angularVelocity = Vector3.zero;
        m_ShortBlockRb.drag = 0.5f;
    }

    public override void OnEpisodeBegin()
    {
        ResetBlock(m_ShortBlockRb);
        transform.localPosition = new Vector3(
            18 * (Random.value - 0.5f), 1, -12);
        m_Goal = Random.Range(0, goals);
        //m_Goal = 0;
        System.Array.Clear(m_GoalOneHot, 0, m_GoalOneHot.Length);
        m_GoalOneHot[m_Goal] = 1;
        if (m_Goal == 0)
        {
            m_WallRenderer.material = goalMaterial;
            m_PushRenderer.material = nonGoalMaterial;
        }
        else
        {
            m_WallRenderer.material = nonGoalMaterial;
            m_PushRenderer.material = goalMaterial;
        }

        var height = m_ResetParams.GetWithDefault("big_wall_height", 4);
        var localScale = wall.transform.localScale;
        localScale = new Vector3(
            localScale.x,
            height,
            localScale.z);
        wall.transform.localScale = localScale;
        m_AgentRb.velocity = default(Vector3);
    }
}
