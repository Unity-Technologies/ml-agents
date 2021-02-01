using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

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
    //float m_LaserLength;
    float m_HitPoints;
    // Speed of agent rotation.
    public float turnSpeed;
    float m_Bonus;

    // Speed of agent movement.
    public float moveSpeed;
    public Material normalMaterial;
    public Material weakMaterial;
    public Material deadMaterial;
    public Laser myLaser;
    public GameObject myBody;


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

    public void MoveAgent(ActionBuffers actionBuffers)
    {
        m_Shoot = false;

        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var continuousActions = actionBuffers.ContinuousActions;
        var discreteActions = actionBuffers.DiscreteActions;

        if (!m_Dead)
        {
            var forward = Mathf.Clamp(continuousActions[0], -1f, 1f);
            var right = Mathf.Clamp(continuousActions[1], -1f, 1f);
            var rotate = Mathf.Clamp(continuousActions[2], -1f, 1f);

            dirToGo = transform.forward * forward;
            dirToGo += transform.right * right;
            rotateDir = -transform.up * rotate;


            var shootCommand = (int)discreteActions[0] > 0;

            if (shootCommand)
            {
                if (Time.time > m_ShootTime + .4f)
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

        if (m_Shoot)
        {
            var myTransform = transform;
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
                    hit.collider.gameObject.GetComponent<LargeCubeAgent>().HitAgent(.02f);

                    AddReward(.1f + .4f * m_Bonus);
                }
                myLaser.isFired = true;
            }
        }
        else if (Time.time > m_ShootTime + .25f)
        {
            myLaser.isFired = false;
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
            m_HitPoints = Mathf.Min(m_HitPoints + .25f, 1f);
            HealthStatus();
        }
    }

    void HealthStatus()
    {
        if (m_HitPoints <= 1f && m_HitPoints > .5f)
        {
            gameObject.tag = "StrongSmallAgent";
            myBody.GetComponentInChildren<Renderer>().material = normalMaterial;
        }

        else if (m_HitPoints <= .5f && m_HitPoints > 0.0f)
        {
            gameObject.tag = "WeakSmallAgent";
            myBody.GetComponentInChildren<Renderer>().material = weakMaterial;

        }
        else // Dead
        {
            AddReward(-.1f * m_Bonus);
            m_Dead = true;
            gameObject.tag = "DeadSmallAgent";
            myBody.GetComponentInChildren<Renderer>().material = deadMaterial;
            m_MyArea.AgentDied();
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        MoveAgent(actionBuffers);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = 0;
        continuousActionsOut[1] = 0;
        continuousActionsOut[2] = 0;
        if (Input.GetKey(KeyCode.D))
        {
            continuousActionsOut[2] = 1;
        }
        if (Input.GetKey(KeyCode.W))
        {
            continuousActionsOut[0] = 1;
        }
        if (Input.GetKey(KeyCode.A))
        {
            continuousActionsOut[2] = -1;
        }
        if (Input.GetKey(KeyCode.S))
        {
            continuousActionsOut[0] = -1;
        }
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = Input.GetKey(KeyCode.Space) ? 1 : 0;
    }

    public override void OnEpisodeBegin()
    {
        m_HitPoints = 1f;
        HealthStatus();
        m_Dead = false;
        m_Shoot = false;
        m_ShootTime = -.5f;
        //m_Bonus = Academy.Instance.FloatProperties.GetPropertyWithDefault("bonus", 0);
        m_Bonus = .2f;//SideChannelUtils.GetSideChannel<FloatPropertiesChannel>().GetPropertyWithDefault("bonus", 0);
        m_AgentRb.velocity = Vector3.zero;

        float smallRange = 50f * m_MyArea.range;
        transform.position = new Vector3(Random.Range(-smallRange, smallRange),
            2f, Random.Range(-smallRange, smallRange))
            + area.transform.position;
        transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));

        SetResetParameters();
    }

    public bool IsDead()
    {
        return m_Dead;
    }

    //public void SetLaserLengths()
    //{
    //    m_LaserLength = 1f;
    //}

    public void SetAgentScale()
    {
        float agentScale = 1f;
        gameObject.transform.localScale = new Vector3(agentScale, agentScale, agentScale);
    }

    public void SetResetParameters()
    {
        //SetLaserLengths();
        SetAgentScale();
    }
}
