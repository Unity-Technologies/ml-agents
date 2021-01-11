//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class DodgeAgent : Agent
{
    /// <summary>
    /// The ground. The bounds are used to spawn the elements.
    /// </summary>
    public GameObject ground;

    public GameObject area;

    /// <summary>
    /// The area bounds.
    /// </summary>
    [HideInInspector]
    public Bounds areaBounds;

    BulletSettings m_BulletSettings;


    Rigidbody m_AgentRb;  //cached on initialization
    Material m_GroundMaterial; //cached on Awake()

    /// <summary>
    /// We will be changing the ground material based on success/failue
    /// </summary>
    Renderer m_GroundRenderer;

    EnvironmentParameters m_ResetParams;

    BufferSensorComponent m_BufferSensor;

    void Awake()
    {
        m_BulletSettings = FindObjectOfType<BulletSettings>();
    }

    public override void Initialize()
    {
        m_BufferSensor = GetComponent<BufferSensorComponent>();

        // Cache the agent rigidbody
        m_AgentRb = GetComponent<Rigidbody>();
        // Cache the block rigidbody
        // Get the ground's bounds
        areaBounds = ground.GetComponent<Collider>().bounds;
        // Get the ground renderer so we can change the material when a goal is scored
        m_GroundRenderer = ground.GetComponent<Renderer>();
        // Starting material
        m_GroundMaterial = m_GroundRenderer.material;

        m_ResetParams = Academy.Instance.EnvironmentParameters;

        SetResetParameters();
    }

    /// <summary>
    /// Use the ground's bounds to pick a random spawn position.
    /// </summary>
    public Vector3 GetRandomSpawnPos()
    {
        var foundNewSpawnLocation = false;
        var randomSpawnPos = Vector3.zero;
        while (foundNewSpawnLocation == false)
        {
            var randomPosX = Random.Range(-areaBounds.extents.x * m_BulletSettings.spawnAreaMarginMultiplier,
                areaBounds.extents.x * m_BulletSettings.spawnAreaMarginMultiplier);

            var randomPosZ = Random.Range(-areaBounds.extents.z * m_BulletSettings.spawnAreaMarginMultiplier,
                areaBounds.extents.z * m_BulletSettings.spawnAreaMarginMultiplier);
            randomSpawnPos = ground.transform.position + new Vector3(randomPosX, 1f, randomPosZ);
            if (Physics.CheckBox(randomSpawnPos, new Vector3(2.5f, 0.01f, 2.5f)) == false)
            {
                foundNewSpawnLocation = true;
            }
        }
        return randomSpawnPos;
    }

    public override void CollectObservations(VectorSensor sensor)
    {

        sensor.AddObservation((transform.position.x - area.transform.position.x) / 10f);
        sensor.AddObservation((transform.position.z - area.transform.position.z) / 10f);

        // Collect observation about the 20 closest Bullets
        var bullets = transform.parent.GetComponentsInChildren<Bullet>();
        // Sort by closest :
        System.Array.Sort(bullets , (a, b) => (Vector3.Distance(a.transform.position, transform.position)).CompareTo(Vector3.Distance(b.transform.position, transform.position)));
        int numBulletAdded = 0;

        // foreach (Bullet b in bullets)
        // {
        //     b.transform.localScale = new Vector3(1, 1, 1);
        // }

        foreach (Bullet b in bullets)
        {
            if (numBulletAdded >= 20){
                break;
            }

            float[] bulletObservation = new float[]{
                (b.transform.position.x - transform.position.x) / 10f, // relative position
                (b.transform.position.z - transform.position.z) / 10f,
                b.transform.forward.x,
                b.transform.forward.z
            };
            numBulletAdded +=1;

            m_BufferSensor.AppendObservation(bulletObservation);

            // b.transform.localScale = new Vector3(2, 2, 2);
        };


    }
    /// <summary>
    /// Called every step of the engine. Here the agent takes an action.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        var forwardForce = Mathf.Clamp(actionBuffers.ContinuousActions[0], -1f, 1f);
        var lateralForce = Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f);
        // var rotationForce = Mathf.Clamp(actionBuffers.ContinuousActions[2], -1f, 1f);

        // Vector3 dirToGo = transform.forward * forwardForce + transform.right * lateralForce;
        // Vector3 rotateDir = transform.up * rotationForce;

        // transform.Rotate(rotateDir, Time.fixedDeltaTime * 200f);
        Vector3 dirToGo = new Vector3(1,0,0) * forwardForce + new Vector3(0,0,1)*lateralForce;
        m_AgentRb.AddForce(dirToGo * m_BulletSettings.agentRunSpeed,
            ForceMode.VelocityChange);
        //Vector3 dirToCenter = new Vector3((transform.position.x - area.transform.position.x) / 10f, 0f, (transform.position.z - area.transform.position.z) / 10f);
        //AddReward(.001f / (dirToCenter.magnitude + .0000001f));
        AddReward(.001f);

    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = 0f;
        continuousActionsOut[1] = 0f;
        continuousActionsOut[2] = 0f;
        if (Input.GetKey(KeyCode.D))
        {
            continuousActionsOut[2] = 1f;
        }
        else if (Input.GetKey(KeyCode.W))
        {
            continuousActionsOut[0] = 1f;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            continuousActionsOut[2] = -1f;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            continuousActionsOut[0] = -1f;
        }
        else if (Input.GetKey(KeyCode.Q))
        {
            continuousActionsOut[1] = -1f;
        }
        else if (Input.GetKey(KeyCode.E))
        {
            continuousActionsOut[1] = 1f;
        }

    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("bullet") || collision.gameObject.CompareTag("wall"))
        {
            SetReward(0f);
            EndEpisode();
        }
    }

    /// <summary>
    /// In the editor, if "Reset On Done" is checked then AgentReset() will be
    /// called automatically anytime we mark done = true in an agent script.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        // var rotation = Random.Range(0, 4);
        // var rotationAngle = rotation * 90f;
        // area.transform.Rotate(new Vector3(0f, rotationAngle, 0f));

        transform.position = GetRandomSpawnPos();//
        m_AgentRb.velocity = Vector3.zero;
        m_AgentRb.angularVelocity = Vector3.zero;

        SetResetParameters();
    }

    public void SetGroundMaterialFriction()
    {
        var groundCollider = ground.GetComponent<Collider>();

        groundCollider.material.dynamicFriction = m_ResetParams.GetWithDefault("dynamic_friction", 0);
        groundCollider.material.staticFriction = m_ResetParams.GetWithDefault("static_friction", 0);
    }


    void SetResetParameters()
    {
        //SetGroundMaterialFriction();
    }
}
