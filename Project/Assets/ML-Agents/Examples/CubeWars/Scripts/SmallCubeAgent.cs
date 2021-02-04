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
    // Speed of agent rotation.

    // Speed of agent movement.
    public Material normalMaterial;
    public Material weakMaterial;
    public Material deadMaterial;
    public Laser myLaser;
    public GameObject myBody;

    float m_HitPoints;
    float m_HitPointsTotal;
    float m_Damage;
    float m_Heal;
    float m_MoveSpeed;
    float m_TurnSpeed;
    float m_Cooldown;
    float m_Range = 25f;
    float m_Splash = 2f;

    public enum Role
    {
        Healer,
        Tank,
        DPS
    }

    public Role role;
    float[] m_RoleObs;

    public override void Initialize()
    {
        m_RoleObs = new float[3];
        if (role == Role.Healer)
        {
            m_RoleObs[0] = 1f;
            m_HitPointsTotal = .5f;
            m_Damage = 0f;
            m_Heal = .2f;
            m_MoveSpeed = 10f;
            m_TurnSpeed = 150f;
            m_Cooldown = .4f;
            m_Splash = 10f;
            m_Range = 15f;
        }
        else if (role == Role.DPS)
        {
            m_RoleObs[1] = 1f;
            m_HitPointsTotal = .5f;
            m_Damage = .05f;
            m_Heal = 0f;
            m_MoveSpeed = 10f;
            m_TurnSpeed = 200f;
            m_Cooldown = .25f;
            m_Splash = 2f;
            m_Range = 25f;
        }
        else if (role == Role.Tank)
        {
            m_RoleObs[2] = 1f;
            m_HitPointsTotal = 1f;
            m_Damage = .02f;
            m_Heal = .1f;
            m_MoveSpeed = 6f;
            m_TurnSpeed = 100f;
            m_Cooldown = .4f;
            m_Splash = 2f;
            m_Range = 25f;
        }

        m_AgentRb = GetComponent<Rigidbody>();
        m_MyArea = area.GetComponent<CubeWarArea>();
        m_LargeAgent = largeAgent.GetComponent<LargeCubeAgent>();
        m_CubeWarSettings = FindObjectOfType<CubeWarSettings>();
        SetResetParameters();
        myLaser.maxLength = m_Range;
        myLaser.width = m_Splash / 2f;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(System.Convert.ToInt32(m_Shoot));
        sensor.AddObservation(m_HitPoints);
        // Direction big agent is looking
        Vector3 dirToSelf = transform.position - m_LargeAgent.transform.position;
        float angle = Vector3.Dot(m_LargeAgent.transform.forward.normalized, dirToSelf.normalized);
        sensor.AddObservation(angle);
        sensor.AddObservation(m_RoleObs);
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
                if (Time.time > m_ShootTime + m_Cooldown)
                {
                    m_Shoot = true;
                    dirToGo *= 0.5f;
                    m_AgentRb.velocity *= 0.9f;
                    m_ShootTime = Time.time;
                }
            }
            transform.Rotate(rotateDir, Time.fixedDeltaTime * m_TurnSpeed);
            m_AgentRb.AddForce(dirToGo * m_MoveSpeed, ForceMode.VelocityChange);
        }

        if (m_Shoot)
        {
            myLaser.isFired = true;
            var myTransform = transform;
            var rayDir = m_Range * myTransform.forward;
            Debug.DrawRay(myTransform.position, rayDir, Color.red, 0f, true);
            RaycastHit hit;
            if (Physics.SphereCast(transform.position, m_Splash, rayDir, out hit, m_Range))
            {
                if (hit.collider.gameObject.CompareTag("StrongSmallAgent") || hit.collider.gameObject.CompareTag("WeakSmallAgent"))
                {
                    hit.collider.gameObject.GetComponent<SmallCubeAgent>().HealAgent(m_Heal);
                    if (role == Role.Healer)
                    {
                        AddReward(.02f);
                    }

                }
                else if (hit.collider.gameObject.CompareTag("StrongLargeAgent") || hit.collider.gameObject.CompareTag("WeakLargeAgent"))
                {
                    hit.collider.gameObject.GetComponent<LargeCubeAgent>().HitAgent(m_Damage);

                    if (role == Role.DPS)
                    {
                        AddReward(.02f);
                    }
                }
            }
        }
        else if (Time.time > m_ShootTime + 0.1f) // This is just how long the graphic stays live
        {
            myLaser.isFired = false;
        }
    }

    public bool HitAgent(float damage) // Returns true if agent dies.
    {
        if (!m_Dead)
        {
            if (role == Role.Tank)
            {
                AddReward(.02f);
            }
            m_HitPoints -= damage;
            HealthStatus();
            return m_HitPoints <= 0;
        }
        else
        {
            return true;
        }
    }

    public void HealAgent(float heal)
    {
        if (m_HitPoints < 1f)
        {
            m_HitPoints = Mathf.Min(m_HitPoints + heal, m_HitPointsTotal);
            HealthStatus();
        }
    }

    void HealthStatus()
    {
        float hitPointRatio = m_HitPoints / m_HitPointsTotal;
        if (hitPointRatio <= 1f && hitPointRatio > .5f)
        {
            gameObject.tag = "StrongSmallAgent";
            myBody.GetComponentInChildren<Renderer>().material = normalMaterial;
        }

        else if (hitPointRatio <= .5f && hitPointRatio > 0.0f)
        {
            gameObject.tag = "WeakSmallAgent";
            myBody.GetComponentInChildren<Renderer>().material = weakMaterial;

        }
        else // Dead
        {
            m_Dead = true;
            gameObject.SetActive(false);
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
        m_HitPoints = m_HitPointsTotal;
        m_Dead = false;
        m_Shoot = false;
        m_ShootTime = -.5f;
        m_AgentRb.velocity = Vector3.zero;
        HealthStatus();

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
