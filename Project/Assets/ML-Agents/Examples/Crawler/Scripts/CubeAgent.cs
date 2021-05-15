using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

using Random = UnityEngine.Random;

public class CubeAgent : Agent
{
    //float m_LateralSpeed;
    //float m_ForwardSpeed;


    [HideInInspector]
    public Rigidbody agentRb;
    BehaviorParameters m_BehaviorParameters;
    public Vector3 initialPos;
    public GameObject area;

    int m_NumDiv = 4;
    EnvironmentParameters m_ResetParams;
    public Transform TargetPrefab; //Target prefab to use in Dynamic envs
    private Transform m_Target; //Target the agent will walk towards during training.

    float[] continuousDiv;
    VectorSensorComponent m_DiversitySettingSensor;
    public bool useContinuous = false;
    public int m_DiversitySetting = 0;
    [Range(-1f, 1f)]
    public float div1;
    [Range(-1f, 1f)]
    public float div2;
    [Range(-1f, 1f)]
    public float div3;
    [Range(-1f, 1f)]
    public float div4;



    public override void Initialize()
    {
        continuousDiv = new float[m_NumDiv];
        SpawnTarget(TargetPrefab, transform.position); //spawn target
        m_BehaviorParameters = gameObject.GetComponent<BehaviorParameters>();
        //m_LateralSpeed = 1.0f;
        //m_ForwardSpeed = 1.0f;
        GetComponent<VectorSensorComponent>().CreateSensors();
        m_DiversitySettingSensor = GetComponent<VectorSensorComponent>();
        agentRb = GetComponent<Rigidbody>();

    }


    void SpawnTarget(Transform prefab, Vector3 pos)
    {
        m_Target = Instantiate(prefab, pos, Quaternion.identity, transform.parent);
    }

    public void MoveAgent(ActionBuffers actionBuffers)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;


        var continuousActions = actionBuffers.ContinuousActions;
        var forward = Mathf.Clamp(continuousActions[0], -1f, 1f);
        var right = Mathf.Clamp(continuousActions[1], -1f, 1f);
        var rotate = Mathf.Clamp(continuousActions[2], -1f, 1f);
        dirToGo = transform.forward * forward;
        dirToGo += transform.right * right;
        rotateDir = -transform.up * rotate;

        transform.Rotate(rotateDir, Time.deltaTime * 100f);
        agentRb.AddForce(dirToGo,
            ForceMode.VelocityChange);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        MoveAgent(actionBuffers);
    }

    void OnCollisionEnter(Collision col)
    {
        if (col.transform.CompareTag("target"))
        {
            SetReward(1f);
            //EndEpisode();
        }

    }

    
    public override void CollectObservations(VectorSensor sensor)
    {
        //AddReward(-1f);
        m_DiversitySettingSensor.GetSensor().Reset();
        if (useContinuous)
        {
            Array.Clear(continuousDiv, 0, m_NumDiv);
            continuousDiv[0] = div1;
            continuousDiv[1] = div2;
            continuousDiv[2] = div3;
            continuousDiv[3] = div4;
            m_DiversitySettingSensor.GetSensor().AddObservation(continuousDiv);
        }
        else
        {
            m_DiversitySettingSensor.GetSensor().AddOneHotObservation(m_DiversitySetting, m_NumDiv);
        }


        //velocity we want to match
        //var velGoal = cubeForward * TargetWalkingSpeed;
        //ragdoll's avg vel
        //var avgVel = GetAvgVelocity();

        //current ragdoll velocity. normalized
        //sensor.AddObservation(Vector3.Distance(velGoal, avgVel));
        //avg body vel relative to cube
        //vel goal relative to cube
        //rotation delta

        //Add pos of target relative to orientation cube
        //sensor.AddObservation(Vector3.Dot(agentRb.velocity, agentRb.transform.forward) / 80f);
        //sensor.AddObservation(Vector3.Dot(agentRb.velocity, agentRb.transform.right) / 80f);
        //sensor.AddObservation(agentRb.transform.forward.x);
        //sensor.AddObservation(agentRb.transform.forward.z);
//        sensor.AddObservation(transform.InverseTransformPoint(m_Target.transform.position).x / 50f);
//        sensor.AddObservation(transform.InverseTransformPoint(m_Target.transform.position).z / 50f);
        //sensor.AddObservation((transform.position.x - m_Target.transform.position.x) / 50f);
        //sensor.AddObservation((transform.position.z - m_Target.transform.position.z) / 50f);
        //sensor.AddObservation(transform.InverseTransformPoint(m_Target.transform.position).z / 50f);

        //sensor.AddObservation((transform.position.x - area.transform.position.x) / 50f);
        //sensor.AddObservation((transform.position.z - area.transform.position.z) / 50f);

    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        if (Input.GetKey(KeyCode.D))
        {
            continuousActionsOut[2] = -1;
        }
        if (Input.GetKey(KeyCode.W))
        {
            continuousActionsOut[0] = 1;
        }
        if (Input.GetKey(KeyCode.A))
        {
            continuousActionsOut[2] = 1;
        }
        if (Input.GetKey(KeyCode.Q))
        {
            continuousActionsOut[1] = -1;
        }
        if (Input.GetKey(KeyCode.E))
        {
            continuousActionsOut[1] = 1;
        }

        if (Input.GetKey(KeyCode.S))
        {
            continuousActionsOut[0] = -1;
        }

           }
    /// <summary>
    /// Used to provide a "kick" to the ball.
    /// </summary>
    
    public override void OnEpisodeBegin()
    {
        if (useContinuous)
        {
            Array.Clear(continuousDiv, 0, m_NumDiv);
            div1 = Random.Range(-1f, 1f);
            div2 = Random.Range(-1f, 1f);
            div3 = Random.Range(-1f, 1f);
            div4 = Random.Range(-1f, 1f);
            continuousDiv[0] = div1;
            continuousDiv[1] = div2;
            continuousDiv[2] = div3;
            continuousDiv[3] = div4;
        }
        else
        {
            m_DiversitySetting = Random.Range(0, m_NumDiv);
        }

        agentRb.velocity = Vector3.zero;
        transform.position = new Vector3(Random.Range(-3f, 3f),
            1f, Random.Range(-3f, 3f))
            + area.transform.position;

        transform.rotation = Quaternion.Euler(new Vector3(0f, Random.Range(0, 360)));
    }


}
