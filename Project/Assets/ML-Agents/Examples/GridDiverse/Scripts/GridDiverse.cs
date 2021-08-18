using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class GridDiverse : Agent
{
    private int state;

    public override void OnEpisodeBegin()
    {
        state = 0;
        transform.localPosition = StateToPosition(state);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddOneHotObservation(state, 4);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        state = actionBuffers.DiscreteActions[0];
        transform.localPosition = StateToPosition(state);
    }

    private static Vector3 StateToPosition(int s)
    {
        switch(s)
        {
            case 0:
                return new Vector3(0, 0, 0);
            case 1:
                return new Vector3(1, 0, 0);
            case 2:
                return new Vector3(1, 0, 1);
            case 3:
                return new Vector3(0, 0, 1);
        }
        return new Vector3(0, 0, 0);
    }
}
