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

    public float m_AgentSpeed = 1;
    public bool m_DenseReward = false;

    protected float lastDist;
    protected float initDist;

    public override void Initialize()
    {
        m_AgentRb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        transform.localPosition = new Vector3(0, 0.225f, 0);
        m_AgentRb.velocity = Vector3.zero;
        m_AgentRb.angularVelocity = Vector3.zero;

        lastDist = GetClosestDist();
        initDist = GetClosestDist();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Vector3.Project(goal1.transform.position - transform.position, 
                                              goal1.transform.forward).magnitude);
        sensor.AddObservation(Vector3.Project(goal2.transform.position - transform.position, 
                                              goal2.transform.right).magnitude);
        sensor.AddObservation(Vector3.Project(goal3.transform.position - transform.position, 
                                              goal3.transform.forward).magnitude);
        sensor.AddObservation(Vector3.Project(goal4.transform.position - transform.position, 
                                              goal4.transform.right).magnitude);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Move(actionBuffers);
        SetStepReward();
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

    protected virtual void Move(ActionBuffers actionBuffers)
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
    }

    protected float GetClosestDist()
    {
        float dist1 = Vector3.Project(goal1.transform.position - transform.position, goal1.transform.forward).magnitude;
        float dist2 = Vector3.Project(goal1.transform.position - transform.position, goal1.transform.forward).magnitude;
        float dist3 = Vector3.Project(goal1.transform.position - transform.position, goal1.transform.forward).magnitude;
        float dist4 = Vector3.Project(goal1.transform.position - transform.position, goal1.transform.forward).magnitude;

        float leastDist = dist1;
        if (dist2 < leastDist)
        {
            leastDist = dist2;
        }
        if (dist3 < leastDist)
        {
            leastDist = dist3;
        }
        if (dist4 < leastDist)
        {
            leastDist = dist4;
        }
        return leastDist;
    }

    protected void SetStepReward()
    {
        AddReward(-1f / MaxStep);
        if (m_DenseReward) 
        {
            float dist = GetClosestDist();
            AddReward((lastDist - dist) / initDist);
            lastDist = dist;
        }
    }
}
