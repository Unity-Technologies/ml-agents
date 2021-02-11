using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class LargeCubeAgent : Agent
{
    CubeWarSettings m_CubeWarSettings;
    public GameObject area;
    CubeWarArea m_MyArea;
    bool m_Dead;
    bool m_Shoot;
    float m_ShootTime;
    bool m_Shockwave;
    float m_ShockwaveTime;
    Rigidbody m_AgentRb;
    //    float m_LaserLength;
    float m_HitPoints;
    // Speed of agent rotation.
    public float turnSpeed;

    // Speed of agent movement.
    public float moveSpeed;

    public float fireDamage = 0.25f;
    public Material normalMaterial;
    public Material weakMaterial;
    public Material deadMaterial;
    public Laser myLaser;
    public GameObject shockwave;
    public GameObject myBody;


    public override void Initialize()
    {
        m_AgentRb = GetComponent<Rigidbody>();
        m_MyArea = area.GetComponent<CubeWarArea>();
        m_CubeWarSettings = FindObjectOfType<CubeWarSettings>();
        SetResetParameters();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(System.Convert.ToInt32(m_Shoot));
        sensor.AddObservation(System.Convert.ToInt32(m_Shockwave));
        sensor.AddObservation(m_HitPoints);
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
        AddReward(-0.0001f);
        m_Shoot = false;
        m_Shockwave = false;

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
            var shockwaveCommand = (int)discreteActions[1] > 0;

            if (shootCommand)
            {
                if (Time.time > m_ShootTime + 1f)
                {
                    m_Shoot = true;
                    dirToGo *= 0.5f;
                    m_AgentRb.velocity *= 0.75f;
                    m_ShootTime = Time.time;
                }
            }
            if (shockwaveCommand)
            {
                if (Time.time > m_ShockwaveTime + 3f)
                {
                    m_Shockwave = true;
                    dirToGo *= 0.5f;
                    m_AgentRb.velocity *= 0.75f;
                    m_ShockwaveTime = Time.time;
                }
            }
            if (!m_Shockwave)
            {
                transform.Rotate(rotateDir, Time.fixedDeltaTime * turnSpeed);
                m_AgentRb.AddForce(dirToGo * moveSpeed, ForceMode.VelocityChange);
            }
        }

        if (m_AgentRb.velocity.sqrMagnitude > 25f) // slow it down
        {
            m_AgentRb.velocity *= 0.95f;
        }

        float checkTime = Time.time;
        if (m_Shoot)
        {
            var myTransform = transform;
            myLaser.isFired = true;
            var rayDir = 120.0f * myTransform.forward;
            Debug.DrawRay(myTransform.position, rayDir, Color.red, 0f, true);
            RaycastHit hit;
            if (Physics.SphereCast(transform.position, 4f, rayDir, out hit, 120f))
            {
                if (hit.collider.gameObject.CompareTag("StrongSmallAgent") || hit.collider.gameObject.CompareTag("WeakSmallAgent"))
                {
                    if (hit.collider.gameObject.GetComponent<SmallCubeAgent>().HitAgent(fireDamage))
                    {
                        AddReward(0.1f);
                    }
                    //AddReward(.1f);
                }
                else if (hit.collider.gameObject.CompareTag("StrongLargeAgent") || hit.collider.gameObject.CompareTag("WeakLargeAgent"))
                {
                    hit.collider.gameObject.GetComponent<LargeCubeAgent>().HealAgent();
                }
            }
        }
        else if (checkTime > m_ShootTime + .5f)
        {
            myLaser.isFired = false;
        }

        if (m_Shockwave)
        {
            // Squish animation
            myBody.transform.localScale = new Vector3(1.2f, 0.8f, 1.2f);
            // Make shockwave animation
            var myTransform = transform;
            shockwave.transform.localScale = new Vector3(1f, 1f, 1f);
            RaycastHit hit;
            int casts = 16;
            float angleRotation = 360f / (float)casts;
            for (int i = 0; i < casts; i++)
            {
                var rayDir = Quaternion.AngleAxis(angleRotation * i, Vector3.up) * myTransform.forward * 10f;
                Debug.DrawRay(myTransform.position, rayDir, Color.green, 0f, true);

                if (Physics.SphereCast(transform.position, 3f, rayDir, out hit, 7f))
                {
                    if (hit.collider.gameObject.CompareTag("StrongSmallAgent") || hit.collider.gameObject.CompareTag("WeakSmallAgent"))
                    {
                        if (hit.collider.gameObject.GetComponent<SmallCubeAgent>().HitAgent(.8f))
                        {
                            AddReward(0.1f);
                        };
                    }
                }
            }
        }
        else if (checkTime > m_ShockwaveTime + 0.3f)
        {
            myBody.transform.localScale = new Vector3(1f, 1f, 1f);
            shockwave.transform.localScale = new Vector3(0f, 0f, 0f);
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
        if (m_HitPoints < 1f && !m_Dead)
        {
            m_HitPoints = Mathf.Min(m_HitPoints + .1f, 1f);
            HealthStatus();
        }
    }

    // Only change material of body and tail
    private void ChangeMaterial(Material material)
    {
        myBody.GetComponentInChildren<Renderer>().material = material;
        GameObject tail = transform.Find("DragonCube").Find("tail").gameObject;
        if (tail != null)
        {
            tail.GetComponentInChildren<Renderer>().material = material;
        }
    }

    public void HealthStatus()
    {
        if (m_HitPoints <= 1f && m_HitPoints > .5f)
        {
            gameObject.tag = "StrongLargeAgent";
            ChangeMaterial(normalMaterial);
        }

        else if (m_HitPoints <= .5f && m_HitPoints > 0.0f)
        {
            gameObject.tag = "WeakLargeAgent";
            ChangeMaterial(weakMaterial);

        }
        else // Dead
        {
            m_Dead = true;
            gameObject.tag = "DeadLargeAgent";
            ChangeMaterial(deadMaterial);
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
        discreteActionsOut[1] = Input.GetKey(KeyCode.O) ? 1 : 0;
    }

    public override void OnEpisodeBegin()
    {
        m_HitPoints = 1f;
        HealthStatus();
        m_Dead = false;
        m_Shoot = false;
        m_ShootTime = -1f;
        m_ShockwaveTime = -3f;
        m_AgentRb.velocity = Vector3.zero;
        shockwave.transform.localScale = new Vector3(0f, 0f, 0f);
        float largeSpawn = 20f * m_MyArea.range;
        transform.position = new Vector3(Random.Range(-largeSpawn, largeSpawn),
            2f, Random.Range(-largeSpawn, largeSpawn))
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
        float agentScale = 5f;
        transform.localScale = new Vector3(agentScale, agentScale, agentScale);
    }

    public void SetResetParameters()
    {
        //    SetLaserLengths();
        SetAgentScale();
    }
}
