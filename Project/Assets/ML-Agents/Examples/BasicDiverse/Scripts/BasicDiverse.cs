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
    public bool m_DenseReward = true;


    public bool ContinuousActions;

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
       // sensor.AddObservation(Vector3.Project(goal1.transform.position - transform.position,
       //                                       goal1.transform.forward).magnitude);
       // sensor.AddObservation(Vector3.Project(goal2.transform.position - transform.position,
       //                                       goal2.transform.right).magnitude);
       // sensor.AddObservation(Vector3.Project(goal3.transform.position - transform.position,
       //                                       goal3.transform.forward).magnitude);
       // sensor.AddObservation(Vector3.Project(goal4.transform.position - transform.position,
       //                                       goal4.transform.right).magnitude);
       // sensor.AddObservation((goal1.transform.position - transform.position).magnitude);
       // sensor.AddObservation((goal2.transform.position - transform.position).magnitude);
       // sensor.AddObservation((goal3.transform.position - transform.position).magnitude);
       // sensor.AddObservation((goal4.transform.position - transform.position).magnitude);
        sensor.AddObservation(transform.localPosition.x);
        sensor.AddObservation(transform.localPosition.z);

        SetStepReward();
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Move(actionBuffers);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        //var discreteActionsOut = actionsOut.DiscreteActions;
        var contActionsOut = actionsOut.ContinuousActions;

        if (Input.GetKey(KeyCode.W))
        {
            contActionsOut[0] = 1f;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            contActionsOut[0] = -1f;
        }
        if (Input.GetKey(KeyCode.D))
        {
            contActionsOut[1] = 1f;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            contActionsOut[1] = -1f;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject == goal1 ||
            other.gameObject == goal2 ||
            other.gameObject == goal3 ||
            other.gameObject == goal4)
        {
            //AddReward(10f);
            EndEpisode();
        }
    }

    protected virtual void Move(ActionBuffers actionBuffers)
    {
        Vector3 dirToGo = Vector3.zero;
        if (ContinuousActions)
        {
            float forwardAction = actionBuffers.ContinuousActions[0];
            float sideAction = actionBuffers.ContinuousActions[1];

            dirToGo += transform.forward * forwardAction;
            dirToGo += transform.right * sideAction;
        }
        else
        {
            int forwardAction = actionBuffers.DiscreteActions[0];
            int sideAction = actionBuffers.DiscreteActions[1];
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
        }
        //dirToGo = Vector3.Normalize(dirToGo);
        m_AgentRb.AddForce(dirToGo * m_AgentSpeed - m_AgentRb.velocity, ForceMode.VelocityChange);
        //m_AgentRb.AddForce(dirToGo * m_AgentSpeed, ForceMode.VelocityChange);

    }

    protected float GetClosestDist()
    {
        float dist1 = (goal1.transform.position - transform.position).magnitude;
        float dist2 = (goal2.transform.position - transform.position).magnitude;
        float dist3 = (goal3.transform.position - transform.position).magnitude;
        float dist4 = (goal4.transform.position - transform.position).magnitude;

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
        //AddReward(-MaxStep / MaxStep);
        //AddReward(-2.0f);
        if (m_DenseReward)
        {
            float dist = GetClosestDist();
            //AddReward((lastDist - dist) / initDist);
            //lastDist = dist;
            //AddReward(.1f * (2.5f - dist));
            //AddReward(-dist);
            AddReward(-1f);
        }
    }
}
