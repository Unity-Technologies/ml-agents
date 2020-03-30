using UnityEngine;
using MLAgents;
using MLAgents.Sensors;
using MLAgents.SideChannels;

public class SmallCubeAgent : Agent
{
    CubeWarSettings m_CubeWarSettings;
    public GameObject area;
    CubeWarArea m_MyArea;
    public GameObject largeAgent;
    LargeCubeAgent m_LargeAgent;
    bool m_Dead;
    bool m_Shoot;
    float m_ShootTime;
    Rigidbody m_AgentRb;
    float m_LaserLength;
    float m_HitPoints;
    // Speed of agent rotation.
    public float turnSpeed;
    float m_Bonus;

    // Speed of agent movement.
    public float moveSpeed;
    public Material normalMaterial;
    public Material weakMaterial;
    public Material deadMaterial;
    public GameObject myLaser;


    public override void Initialize()
    {
        m_AgentRb = GetComponent<Rigidbody>();
        m_MyArea = area.GetComponent<CubeWarArea>();
        m_LargeAgent = largeAgent.GetComponent<LargeCubeAgent>();
        m_CubeWarSettings = FindObjectOfType<CubeWarSettings>();
        SetResetParameters();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(System.Convert.ToInt32(m_Shoot));
        sensor.AddObservation(System.Convert.ToInt32(m_Dead));
        sensor.AddObservation(m_HitPoints);
        // Direction big agent is looking
        Vector3 dirToSelf = transform.position - m_LargeAgent.transform.position;
        float angle = Vector3.Dot(m_LargeAgent.transform.forward.normalized, dirToSelf.normalized);
        sensor.AddObservation(angle);
        if (m_Dead)
        {
            AddReward(-.001f * m_Bonus);
        }
    }

    public Color32 ToColor(int hexVal)
    {
        var r = (byte)((hexVal >> 16) & 0xFF);
        var g = (byte)((hexVal >> 8) & 0xFF);
        var b = (byte)(hexVal & 0xFF);
        return new Color32(r, g, b, 255);
    }

    public void MoveAgent(float[] act)
    {
        m_Shoot = false;

        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        if (!m_Dead)
        {
            var shootCommand = false;
            var forwardAxis = (int)act[0];
            var rightAxis = (int)act[1];
            var rotateAxis = (int)act[2];
            var shootAxis = (int)act[3];

            switch (forwardAxis)
            {
                case 1:
                    dirToGo = transform.forward;
                    break;
                case 2:
                    dirToGo = -transform.forward;
                    break;
            }

            switch (rightAxis)
            {
                case 1:
                    dirToGo = transform.right;
                    break;
                case 2:
                    dirToGo = -transform.right;
                    break;
            }

            switch (rotateAxis)
            {
                case 1:
                    rotateDir = -transform.up;
                    break;
                case 2:
                    rotateDir = transform.up;
                    break;
            }
            switch (shootAxis)
            {
                case 1:
                    shootCommand = true;
                    break;
            }
            if (shootCommand)
            {
                if (Time.time > m_ShootTime + .5f)
                {
                    m_Shoot = true;
                    dirToGo *= 0.5f;
                    m_AgentRb.velocity *= 0.9f;
                    m_ShootTime = Time.time;
                }
            }
            transform.Rotate(rotateDir, Time.fixedDeltaTime * turnSpeed);
            m_AgentRb.AddForce(dirToGo * moveSpeed, ForceMode.VelocityChange);
        }

        //if (m_AgentRb.velocity.sqrMagnitude > 25f) // slow it down
        //{
        //    m_AgentRb.velocity *= 0.95f;
        //}

        if (m_Shoot)
        {
            var myTransform = transform;
            myLaser.transform.localScale = new Vector3(1f, 1f, m_LaserLength);
            var rayDir = 25.0f * myTransform.forward;
            Debug.DrawRay(myTransform.position, rayDir, Color.red, 0f, true);
            RaycastHit hit;
            if (Physics.SphereCast(transform.position, 2f, rayDir, out hit, 28f))
            {
                if (hit.collider.gameObject.CompareTag("StrongSmallAgent") || hit.collider.gameObject.CompareTag("WeakSmallAgent"))
                {
                    hit.collider.gameObject.GetComponent<SmallCubeAgent>().HealAgent();
                }
                else if (hit.collider.gameObject.CompareTag("StrongLargeAgent") || hit.collider.gameObject.CompareTag("WeakLargeAgent"))
                {
                    hit.collider.gameObject.GetComponent<LargeCubeAgent>().HitAgent(.05f);

                    AddReward(.5f * m_Bonus);
                }
            }
        }
        else if (Time.time > m_ShootTime + .25f)
        {
            myLaser.transform.localScale = new Vector3(0f, 0f, 0f);
        }
    }

    public void HitAgent(float damage)
    {
        if (!m_Dead)
        {
            m_HitPoints -= damage;
            HealthStatus();
        }
    }

    public void HealAgent()
    {
        if (m_HitPoints < 1f)
        {
            m_HitPoints = Mathf.Min(m_HitPoints + .1f, 1f);
            HealthStatus();
        }
    }

    void HealthStatus()
    {
        if (m_HitPoints <= 1f && m_HitPoints > .65f)
        {
            gameObject.tag = "StrongSmallAgent";
            gameObject.GetComponentInChildren<Renderer>().material = normalMaterial;
        }

        else if (m_HitPoints <= .65f && m_HitPoints > .3f)
        {
            gameObject.tag = "WeakSmallAgent";
            gameObject.GetComponentInChildren<Renderer>().material = weakMaterial;

        }
        else // Dead
        {
            AddReward(-.1f * m_Bonus);
            m_Dead = true;
            gameObject.tag = "DeadSmallAgent";
            gameObject.GetComponentInChildren<Renderer>().material = deadMaterial;
            m_MyArea.AgentDied();
        }
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        MoveAgent(vectorAction);
    }

    public override float[] Heuristic()
    {
        var action = new float[4];
        if (Input.GetKey(KeyCode.D))
        {
            action[2] = 2f;
        }
        if (Input.GetKey(KeyCode.W))
        {
            action[0] = 1f;
        }
        if (Input.GetKey(KeyCode.E))
        {
            action[1] = 1f;
        }
        if (Input.GetKey(KeyCode.Q))
        {
            action[1] = 2f;
        }
        if (Input.GetKey(KeyCode.A))
        {
            action[2] = 1f;
        }
        if (Input.GetKey(KeyCode.S))
        {
            action[0] = 2f;
        }
        action[3] = Input.GetKey(KeyCode.Space) ? 1.0f : 0.0f;
        return action;
    }

    public override void OnEpisodeBegin()
    {
        m_HitPoints = 1f;
        HealthStatus();
        m_Dead = false;
        m_Shoot = false;
        m_ShootTime = -.5f;
        //m_Bonus = Academy.Instance.FloatProperties.GetPropertyWithDefault("bonus", 0);
        m_Bonus = SideChannelUtils.GetSideChannel<FloatPropertiesChannel>().GetPropertyWithDefault("bonus", 0);
        m_AgentRb.velocity = Vector3.zero;
        myLaser.transform.localScale = new Vector3(0f, 0f, 0f);
        float smallRange = 50f * m_MyArea.range;
        transform.position = new Vector3(Random.Range(-smallRange, smallRange),
            2f,Random.Range(-smallRange, smallRange))
            + area.transform.position;
        transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));

        SetResetParameters();
    }

    public bool IsDead()
    {
        return m_Dead;
    }

    public void SetLaserLengths()
    {
        m_LaserLength = 1f;
    }

    public void SetAgentScale()
    {
        float agentScale = 1f;
        gameObject.transform.localScale = new Vector3(agentScale, agentScale, agentScale);
    }

    public void SetResetParameters()
    {
        SetLaserLengths();
        SetAgentScale();
    }
}
