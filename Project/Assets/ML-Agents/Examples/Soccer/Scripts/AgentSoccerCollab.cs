using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;

public class AgentSoccerCollab : AgentSoccer
{
    public int tester = 0;
    int m_Previous = 0;


    float[] m_Message = new float[2];
    public GameObject teammate_gb;
    AgentSoccerCollab teammate;

    public override void Initialize()
    {
        base.Initialize();
        teammate = teammate_gb.GetComponent<AgentSoccerCollab>();
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        base.OnActionReceived(actionBuffers);
        //if (team == Team.Blue && tester == 0)
        //{
        //    Debug.Log("cont");
        //    Debug.Log(actionBuffers.ContinuousActions[2]);
        //}
        //else if (team == Team.Blue && tester == 1)
        //{
        //    Debug.Log("mess");
        //    Debug.Log(actionBuffers.DiscreteActions[0]);
        //}
        teammate.tellAgent(actionBuffers.DiscreteActions[0]);
    }

    public override void CollectObservations(VectorSensor sensor)
    {

        sensor.AddObservation(m_Message);
    }

    public void tellAgent(int message)
    {
        m_Message[m_Previous] = 0f;
        if (team == Team.Purple)
        {
            message = 0;//Random.Range(0, 2);
        }
        m_Message[message] = 1f;
        m_Previous = message;
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
        System.Array.Clear(m_Message, 0, m_Message.Length);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut.Clear();
        if (Input.GetKey(KeyCode.Alpha0))
        {
            discreteActionsOut[0] = 0;
        }
        if (Input.GetKey(KeyCode.Alpha1))
        {
            discreteActionsOut[0] = 1;
        }
        if (Input.GetKey(KeyCode.Alpha2))
        {
            discreteActionsOut[0] = 2;
        }
        if (Input.GetKey(KeyCode.Alpha3))
        {
            discreteActionsOut[0] = 3;
        }
        if (Input.GetKey(KeyCode.Alpha4))
        {
            discreteActionsOut[0] = 4;
        }
        if (Input.GetKey(KeyCode.Alpha5))
        {
            discreteActionsOut[0] = 5;
        }
        if (Input.GetKey(KeyCode.Alpha6))
        {
            discreteActionsOut[0] = 6;
        }
        if (Input.GetKey(KeyCode.Alpha7))
        {
            discreteActionsOut[0] = 7;
        }
        if (Input.GetKey(KeyCode.Alpha8))
        {
            discreteActionsOut[0] = 8;
        }
        if (Input.GetKey(KeyCode.Alpha9))
        {
            discreteActionsOut[0] = 9;
        }

        var contOut = actionsOut.ContinuousActions;
        contOut.Clear();
        //forward
        if (Input.GetKey(KeyCode.W))
        {
            contOut[0] = 1f;
        }
        if (Input.GetKey(KeyCode.S))
        {
            contOut[0] = -1f;
        }
        //rotate
        if (Input.GetKey(KeyCode.Q))
        {
            contOut[1] = -1f;
        }
        if (Input.GetKey(KeyCode.E))
        {
            contOut[1] = 1f;
        }
        //right
        if (Input.GetKey(KeyCode.D))
        {
            contOut[2] = 1f;
        }
        if (Input.GetKey(KeyCode.A))
        {
            contOut[2] = -1f;
        }

        ////forward
        //if (Input.GetKey(KeyCode.W))
        //{
        //    discreteActionsOut[0] = 1;
        //}
        //if (Input.GetKey(KeyCode.S))
        //{
        //    discreteActionsOut[0] = 2;
        //}
        ////rotate
        //if (Input.GetKey(KeyCode.A))
        //{
        //    discreteActionsOut[2] = 1;
        //}
        //if (Input.GetKey(KeyCode.D))
        //{
        //    discreteActionsOut[2] = 2;
        //}
        ////right
        //if (Input.GetKey(KeyCode.E))
        //{
        //    discreteActionsOut[1] = 1;
        //}
        //if (Input.GetKey(KeyCode.Q))
        //{
        //    discreteActionsOut[1] = 2;
        //}
    }
}
