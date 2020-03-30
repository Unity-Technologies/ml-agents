using UnityEngine;
using MLAgents;
using MLAgents.Sensors;

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
    float m_LaserLength;
    float m_HitPoints;
    // Speed of agent rotation.
    public float turnSpeed;

    // Speed of agent movement.
    public float moveSpeed;
    public Material normalMaterial;
    public Material weakMaterial;
    public Material deadMaterial;
    public GameObject myLaser;
    public GameObject shockwave;


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
        sensor.AddObservation(m_HitPoints);
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
        m_Shockwave = false;

        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        if (!m_Dead)
        {
            var shootCommand = false;
            var shockwaveCommand = false;
            var forwardAxis = (int)act[0];
            var rightAxis = (int)act[1];
            var rotateAxis = (int)act[2];
            var shootAxis = (int)act[3];
            var shockwaveAxis = (int)act[4];

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
            switch (shockwaveAxis)
            {
                case 1:
                    shockwaveCommand = true;
                    break;
            }
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
            transform.Rotate(rotateDir, Time.fixedDeltaTime * turnSpeed);
            m_AgentRb.AddForce(dirToGo * moveSpeed, ForceMode.VelocityChange);
        }

        if (m_AgentRb.velocity.sqrMagnitude > 25f) // slow it down
        {
            m_AgentRb.velocity *= 0.95f;
        }

        float checkTime = Time.time;
        if (m_Shoot)
        {
            var myTransform = transform;
            myLaser.transform.localScale = new Vector3(1f, 1f, m_LaserLength);
            var rayDir = 120.0f * myTransform.forward;
            Debug.DrawRay(myTransform.position, rayDir, Color.red, 0f, true);
            RaycastHit hit;
            if (Physics.SphereCast(transform.position, 7f, rayDir, out hit, 120f))
            {
                if (hit.collider.gameObject.CompareTag("StrongSmallAgent") || hit.collider.gameObject.CompareTag("WeakSmallAgent"))
                {
                    hit.collider.gameObject.GetComponent<SmallCubeAgent>().HitAgent(.35f);
                }
                else if (hit.collider.gameObject.CompareTag("StrongLargeAgent") || hit.collider.gameObject.CompareTag("WeakLargeAgent"))
                {
                    hit.collider.gameObject.GetComponent<LargeCubeAgent>().HealAgent();
                }
            }
        }
        else if (checkTime > m_ShootTime + .5f)
        {
            myLaser.transform.localScale = new Vector3(0f, 0f, 0f);
        }

        if (m_Shockwave)
        {
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
                        hit.collider.gameObject.GetComponent<SmallCubeAgent>().HitAgent(1f);
                    }
                }
            }
        }
        else if (checkTime > m_ShockwaveTime + 1.5f)
        {
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

    public void HealthStatus()
    {
        if (m_HitPoints <= 1f && m_HitPoints > .5f)
        {
            gameObject.tag = "StrongLargeAgent";
            gameObject.GetComponentInChildren<Renderer>().material = normalMaterial;
        }

        else if (m_HitPoints <= .5f && m_HitPoints > 0.0f)
        {
            gameObject.tag = "WeakLargeAgent";
            gameObject.GetComponentInChildren<Renderer>().material = weakMaterial;

        }
        else // Dead
        {
            m_Dead = true;
            gameObject.tag = "DeadLargeAgent";
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
        var action = new float[5];
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
        action[4] = Input.GetKey(KeyCode.O) ? 1.0f : 0.0f;
        return action;
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
        myLaser.transform.localScale = new Vector3(0f, 0f, 0f);
        shockwave.transform.localScale = new Vector3(0f, 0f, 0f);
        transform.position = new Vector3(Random.Range(-m_MyArea.range, m_MyArea.range),
            2f, Random.Range(-m_MyArea.range, m_MyArea.range))
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
        m_LaserLength = Academy.Instance.FloatProperties.GetPropertyWithDefault("laser_length", 1.0f);
    }

    public void SetAgentScale()
    {
        float agentScale = Academy.Instance.FloatProperties.GetPropertyWithDefault("agent_scale", 5.0f);
        gameObject.transform.localScale = new Vector3(agentScale, agentScale, agentScale);
    }

    public void SetResetParameters()
    {
        SetLaserLengths();
        SetAgentScale();
    }
}
