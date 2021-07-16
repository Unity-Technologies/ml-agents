//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class BasicDiverseRotate : BasicDiverse
{
    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();

        float startRot = Random.Range(0f, 360f);
        transform.localEulerAngles = new Vector3(0f, startRot, 0f);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        base.CollectObservations(sensor);
        sensor.AddObservation(transform.localEulerAngles.y / 360f);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        int action = actionBuffers.DiscreteActions[0];

        Vector3 dirToGo = Vector3.zero;
        Vector3 rotateDir = Vector3.zero;
        switch (action)
        {
            case 1:
                dirToGo += transform.forward;
                break;
            case 2:
                rotateDir += transform.up;
                break;
            case 3:
                rotateDir -= transform.up;
                break;
        }

        transform.Rotate(rotateDir, Time.fixedDeltaTime * 200f);
        m_AgentRb.AddForce(dirToGo * m_AgentSpeed, ForceMode.VelocityChange);

        AddReward(-1f / MaxStep);
        if (m_DenseReward) 
        {
            float dist = GetClosestDist();
            AddReward((lastDist - dist) / initDist);
            lastDist = dist;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;

        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 2;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 3;
        }
    }
}
