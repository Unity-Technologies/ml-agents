using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using MLAgents.Sensor;

public class ReacherDecision : Decision
{
    public override float[] Decide(List<float> state, List<ISensor> sensors, float reward, bool done, List<float> memory)
    {
        var action = new float[4];
        for (var i = 0; i < 4; i++)
        {
            action[i] = Random.Range(-1f, 1f);
        }
        return action;
    }

    public override List<float> MakeMemory(List<float> state, List<ISensor> sensors, float reward, bool done, List<float> memory)
    {
        return new List<float>();
    }
}
