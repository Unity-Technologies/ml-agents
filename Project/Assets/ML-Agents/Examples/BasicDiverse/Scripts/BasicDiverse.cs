//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class BasicDiverse : Agent
{

    public GameObject goal1;
    public GameObject goal2;
    public GameObject goal3;
    public GameObject goal4;
    protected Rigidbody m_AgentRb;

    VectorSensorComponent m_DiversitySettingSensor;
    public int m_DiversitySetting = 0;
    public int m_NumDiversityBehaviors = 4;
    public float m_AgentSpeed = 1;

    public override void Initialize()
    {
        m_AgentRb = GetComponent<Rigidbody>();

        m_DiversitySettingSensor = GetComponent<VectorSensorComponent>();
        m_DiversitySettingSensor.CreateSensors();
    }

    public override void OnEpisodeBegin()
    {
        transform.localPosition = new Vector3(0, 0.225f, 0);
        m_AgentRb.velocity = Vector3.zero;
        m_AgentRb.angularVelocity = Vector3.zero;

        m_DiversitySetting = Random.Range(0, m_NumDiversityBehaviors);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        m_DiversitySettingSensor.GetSensor().Reset();
        m_DiversitySettingSensor.GetSensor().AddOneHotObservation(m_DiversitySetting, m_NumDiversityBehaviors);

        sensor.AddObservation(Vector3.Project(goal1.transform.position - transform.position, 
                                              goal1.transform.forward).magnitude);
        sensor.AddObservation(Vector3.Project(goal2.transform.position - transform.position, 
                                              goal2.transform.forward).magnitude);
        sensor.AddObservation(Vector3.Project(goal3.transform.position - transform.position, 
                                              goal3.transform.forward).magnitude);
        sensor.AddObservation(Vector3.Project(goal4.transform.position - transform.position, 
                                              goal4.transform.forward).magnitude);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        int forwardAction = actionBuffers.DiscreteActions[0];
        int sideAction = actionBuffers.DiscreteActions[1];

        Vector3 dirToGo = Vector3.zero;
        switch (forwardAction)
        {
            case 1:
                dirToGo += transform.forward;
                break;
            case 2:
                dirToGo -= transform.forward;
                break;
        }
        switch (sideAction)
        {
            case 1:
                dirToGo += transform.right;
                break;
            case 2:
                dirToGo -= transform.right;
                break;
        }

        dirToGo = Vector3.Normalize(dirToGo);
        m_AgentRb.AddForce(dirToGo * m_AgentSpeed, ForceMode.VelocityChange);
        AddReward(-1f / MaxStep);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;

        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[1] = 1;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[1] = 2;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject == goal1 ||
            other.gameObject == goal2 ||
            other.gameObject == goal3 ||
            other.gameObject == goal4)
        {
            AddReward(1f);
            EndEpisode();
        }
    }
}
