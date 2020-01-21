using UnityEngine;
using MLAgents;

public class FoodCollectorAgent : Agent
{
    FoodCollectorSettings m_FoodCollecterSettings;
    public GameObject area;
    FoodCollectorArea m_MyArea;
    bool m_Frozen;
    bool m_Poisoned;
    bool m_Satiated;
    bool m_Shoot;
    float m_FrozenTime;
    float m_EffectTime;
    Rigidbody m_AgentRb;
    float m_LaserLength;
    // Speed of agent rotation.
    public float turnSpeed = 300;

    // Speed of agent movement.
    public float moveSpeed = 2;
    public Material normalMaterial;
    public Material badMaterial;
    public Material goodMaterial;
    public Material frozenMaterial;
    public GameObject myLaser;
    public bool contribute;
    public bool useVectorObs;


    public override void InitializeAgent()
    {
        base.InitializeAgent();
        m_AgentRb = GetComponent<Rigidbody>();
        Monitor.verticalOffset = 1f;
        m_MyArea = area.GetComponent<FoodCollectorArea>();
        m_FoodCollecterSettings = FindObjectOfType<FoodCollectorSettings>();

        SetResetParameters();
    }

    public override void CollectObservations()
    {
        if (useVectorObs)
        {
            var localVelocity = transform.InverseTransformDirection(m_AgentRb.velocity);
            AddVectorObs(localVelocity.x);
            AddVectorObs(localVelocity.z);
            AddVectorObs(System.Convert.ToInt32(m_Frozen));
            AddVectorObs(System.Convert.ToInt32(m_Shoot));
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

        if (Time.time > m_FrozenTime + 4f && m_Frozen)
        {
            Unfreeze();
        }
        if (Time.time > m_EffectTime + 0.5f)
        {
            if (m_Poisoned)
            {
                Unpoison();
            }
            if (m_Satiated)
            {
                Unsatiate();
            }
        }

        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        if (!m_Frozen)
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
                m_Shoot = true;
                dirToGo *= 0.5f;
                m_AgentRb.velocity *= 0.75f;
            }
            m_AgentRb.AddForce(dirToGo * moveSpeed, ForceMode.VelocityChange);
            transform.Rotate(rotateDir, Time.fixedDeltaTime * turnSpeed);
        }

        if (m_AgentRb.velocity.sqrMagnitude > 25f) // slow it down
        {
            m_AgentRb.velocity *= 0.95f;
        }

        if (m_Shoot)
        {
            var myTransform = transform;
            myLaser.transform.localScale = new Vector3(1f, 1f, m_LaserLength);
            var rayDir = 25.0f * myTransform.forward;
            Debug.DrawRay(myTransform.position, rayDir, Color.red, 0f, true);
            RaycastHit hit;
            if (Physics.SphereCast(transform.position, 2f, rayDir, out hit, 25f))
            {
                if (hit.collider.gameObject.CompareTag("agent"))
                {
                    hit.collider.gameObject.GetComponent<FoodCollectorAgent>().Freeze();
                }
            }
        }
        else
        {
            myLaser.transform.localScale = new Vector3(0f, 0f, 0f);
        }
    }

    void Freeze()
    {
        gameObject.tag = "frozenAgent";
        m_Frozen = true;
        m_FrozenTime = Time.time;
        gameObject.GetComponentInChildren<Renderer>().material = frozenMaterial;
    }

    void Unfreeze()
    {
        m_Frozen = false;
        gameObject.tag = "agent";
        gameObject.GetComponentInChildren<Renderer>().material = normalMaterial;
    }

    void Poison()
    {
        m_Poisoned = true;
        m_EffectTime = Time.time;
        gameObject.GetComponentInChildren<Renderer>().material = badMaterial;
    }

    void Unpoison()
    {
        m_Poisoned = false;
        gameObject.GetComponentInChildren<Renderer>().material = normalMaterial;
    }

    void Satiate()
    {
        m_Satiated = true;
        m_EffectTime = Time.time;
        gameObject.GetComponentInChildren<Renderer>().material = goodMaterial;
    }

    void Unsatiate()
    {
        m_Satiated = false;
        gameObject.GetComponentInChildren<Renderer>().material = normalMaterial;
    }

    public override void AgentAction(float[] vectorAction)
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

    public override void AgentReset()
    {
        Unfreeze();
        Unpoison();
        Unsatiate();
        m_Shoot = false;
        m_AgentRb.velocity = Vector3.zero;
        myLaser.transform.localScale = new Vector3(0f, 0f, 0f);
        transform.position = new Vector3(Random.Range(-m_MyArea.range, m_MyArea.range),
            2f, Random.Range(-m_MyArea.range, m_MyArea.range))
            + area.transform.position;
        transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));

        SetResetParameters();
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("food"))
        {
            Satiate();
            collision.gameObject.GetComponent<FoodLogic>().OnEaten();
            AddReward(1f);
            if (contribute)
            {
                m_FoodCollecterSettings.totalScore += 1;
            }
        }
        if (collision.gameObject.CompareTag("badFood"))
        {
            Poison();
            collision.gameObject.GetComponent<FoodLogic>().OnEaten();

            AddReward(-1f);
            if (contribute)
            {
                m_FoodCollecterSettings.totalScore -= 1;
            }
        }
    }

    public void SetLaserLengths()
    {
        m_LaserLength = Academy.Instance.FloatProperties.GetPropertyWithDefault("laser_length", 1.0f);
    }

    public void SetAgentScale()
    {
        float agentScale = Academy.Instance.FloatProperties.GetPropertyWithDefault("agent_scale", 1.0f);
        gameObject.transform.localScale = new Vector3(agentScale, agentScale, agentScale);
    }

    public void SetResetParameters()
    {
        SetLaserLengths();
        SetAgentScale();
    }
}
