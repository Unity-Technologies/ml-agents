//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using MLAgents;
using Barracuda;


public class WallJump2DAgent : Agent
{
    // Depending on this value, the wall will have different height
    int m_Configuration;

    public GameObject ground;
    public GameObject spawnArea;
    Bounds m_SpawnAreaBounds;


    public GameObject goal;
    public GameObject shortBlock;
    public GameObject wall;
    Rigidbody2D m_AgentRb;
    Material m_GroundMaterial;
    Renderer m_GroundRenderer;
    WallJump2DAcademy m_Academy;

    public float jumpingTime;
    public float jumpTime;
    // This is a downward force applied when falling to make jumps look
    // less floaty
    public float fallingForce;
    // Use to check the coliding objects
    public Collider2D[] hitGroundColliders = new Collider2D[3];
    Vector3 m_JumpTargetPos;
    Vector3 m_JumpStartingPos;

    public override void InitializeAgent()
    {
        m_Academy = FindObjectOfType<WallJump2DAcademy>();
        m_Configuration = Random.Range(0, 5);

        m_AgentRb = GetComponent<Rigidbody2D>();
        m_SpawnAreaBounds = spawnArea.GetComponent<Collider2D>().bounds;
        m_GroundRenderer = ground.GetComponent<Renderer>();
        m_GroundMaterial = m_GroundRenderer.material;

        spawnArea.SetActive(false);
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
        return true;
//        if (!smallCheck)
//        {
//            hitGroundColliders = new Collider2D[3];
//            var o = gameObject;
//            Physics2D.OverlapBoxNonAlloc(
//                o.transform.position + new Vector3(0, -0.05f, 0),
//                new Vector3(0.95f / 2f, 0.5f, 0.95f / 2f),
//                hitGroundColliders,
//                o.transform.rotation);
//            var grounded = false;
//            foreach (var col in hitGroundColliders)
//            {
//                if (col != null && col.transform != transform &&
//                    (col.CompareTag("walkableSurface") ||
//                     col.CompareTag("block") ||
//                     col.CompareTag("wall")))
//                {
//                    grounded = true; //then we're grounded
//                    break;
//                }
//            }
//            return grounded;
//        }
//        else
        {
            RaycastHit2D hit = Physics2D.Raycast(
                (Vector2)transform.position + new Vector2(0, -0.05f),
                -Vector2.up,
                1f
            );

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
        Vector2 targetPos, Rigidbody2D rb, float targetVel, float maxVel)
    {
        var moveToPos = targetPos - rb.worldCenterOfMass;
        var velocityTarget = Time.fixedDeltaTime * targetVel * moveToPos;
        if (float.IsNaN(velocityTarget.x) == false)
        {
            rb.velocity = Vector2.MoveTowards(rb.velocity, velocityTarget, maxVel);
        }
    }

    public override void CollectObservations()
    {
        var agentPos = m_AgentRb.position - (Vector2) ground.transform.position;

        AddVectorObs(agentPos / 20f);
        //AddVectorObs(DoGroundCheck(true) ? 1 : 0);
    }

    /// <summary>
    /// Gets a random spawn position in the spawningArea.
    /// </summary>
    /// <returns>The random spawn position.</returns>
    public Vector2 GetRandomSpawnPos()
    {
        var randomPosX = Random.Range(-m_SpawnAreaBounds.extents.x,
            m_SpawnAreaBounds.extents.x);
        var randomPosY = Random.Range(-m_SpawnAreaBounds.extents.y,
            m_SpawnAreaBounds.extents.y);
        var transformPos = (Vector2) spawnArea.transform.position;
        var randomSpawnPos = transformPos + new Vector2(randomPosX, randomPosY);
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
        yield return new WaitForSeconds(time); //wait for 2 sec
        m_GroundRenderer.material = m_GroundMaterial;
    }

    public void MoveAgent(float[] act)
    {
        AddReward(-0.0005f);
        var smallGrounded = DoGroundCheck(true);
        var largeGrounded = DoGroundCheck(false);

        var dirToGo = Vector2.zero;
        var dirToGoSideAction = (int)act[0];
        var jumpAction = (int)act[1];

        if (dirToGoSideAction != 0)
        {
            Debug.Log($"dirToGoSideAction={dirToGoSideAction}, transform.right={transform.right} transform.up={transform.up} transform.fwd={transform.forward}");
        }
        if (dirToGoSideAction == 1)
            dirToGo = (largeGrounded ? 1f : 0.5f) * -0.6f * transform.right;
        else if (dirToGoSideAction == 2)
            dirToGo = (largeGrounded ? 1f : 0.5f) * 0.6f * transform.right;
        if (jumpAction == 1)
            if ((jumpingTime <= 0f) && smallGrounded)
            {
                Jump();
            }

        //transform.Rotate(rotateDir, Time.fixedDeltaTime * 300f);
        m_AgentRb.AddForce(dirToGo * m_Academy.agentRunSpeed,
            ForceMode2D.Impulse);

        if (jumpingTime > 0f)
        {
            m_JumpTargetPos =
                new Vector2(m_AgentRb.position.x,
                    m_JumpStartingPos.y + m_Academy.agentJumpHeight) + dirToGo;
            MoveTowards(m_JumpTargetPos, m_AgentRb, m_Academy.agentJumpVelocity,
                m_Academy.agentJumpVelocityMaxChange);
        }

        if (!(jumpingTime > 0f) && !largeGrounded)
        {
            m_AgentRb.AddForce(
                Vector3.down * fallingForce, ForceMode2D.Force);
        }
        jumpingTime -= Time.fixedDeltaTime;
    }

    public override void AgentAction(float[] vectorAction)
    {
        MoveAgent(vectorAction);
        // TODO can't happen in 2D?
        if (!Physics2D.Raycast(m_AgentRb.position, Vector2.down, 20))
        {
            Done();
            SetReward(-1f);
            StartCoroutine(
                GoalScoredSwapGroundMaterial(m_Academy.failMaterial, .5f));
        }
    }

    public override float[] Heuristic()
    {
        GenerateSensorData();
        bool pressed = false;
        var action = new float[4];
        if (Input.GetKey(KeyCode.D))
        {
            pressed = true;
            action[0] = 2f;
        }
        if (Input.GetKey(KeyCode.A))
        {
            pressed = true;
            action[0] = 1f;
        }

        if (Input.GetKey(KeyCode.Space))
        {
            pressed = true;
            action[1] = 1.0f;
        }

        if (pressed)
        {
            Debug.Log($"({action[0]}, {action[1]}, {action[2]}, {action[3]})");
        }

        return action;
    }

    // Detect when the agent hits the goal
    void OnTriggerStay2D(Collider2D col)
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
//    void ResetBlock(Rigidbody blockRb)
//    {
//        blockRb.transform.position = GetRandomSpawnPos();
//        blockRb.velocity = Vector3.zero;
//        blockRb.angularVelocity = Vector3.zero;
//    }

    public override void AgentReset()
    {
        Debug.Log("AgentReset");
        transform.localPosition = new Vector3(4 * (Random.value - 0.5f), 1, -2);
        //var randomPos2d = GetRandomSpawnPos();
        //var randomPos = new Vector3(randomPos2d.x, randomPos2d.y, transform.localPosition.z);
        //transform.localPosition = randomPos;
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
    /// Other : Tall wall and BigWallBrain. </param>
    void ConfigureAgent(int config)
    {
        if (wall == null)
        {
            // TODO handle reset here
            return;
        }
        var localScale = wall.transform.localScale;
        if (config == 0)
        {
            localScale = new Vector3(
                localScale.x,
                m_Academy.FloatProperties.GetPropertyWithDefault("no_wall_height", 0),
                localScale.z);
            wall.transform.localScale = localScale;
        }
        else if (config == 1)
        {
            localScale = new Vector3(
                localScale.x,
                m_Academy.FloatProperties.GetPropertyWithDefault("small_wall_height", 4),
                localScale.z);
            wall.transform.localScale = localScale;
        }
        else
        {
            var min = m_Academy.FloatProperties.GetPropertyWithDefault("big_wall_min_height", 8);
            var max = m_Academy.FloatProperties.GetPropertyWithDefault("big_wall_max_height", 8);
            var height = min + Random.value * (max - min);
            localScale = new Vector3(
                localScale.x,
                height,
                localScale.z);
            wall.transform.localScale = localScale;
        }
    }
}
