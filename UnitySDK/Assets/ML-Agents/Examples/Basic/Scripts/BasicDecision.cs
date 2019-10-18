using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using MLAgents.Sensor;

public class BasicDecision : Decision
{
    public override float[] Decide(
        List<float> vectorObs,
        List<ISensor> sensors,
        float reward,
        bool done,
        List<float> memory)
    {
        return new[] { 1f };
    }

    public override List<float> MakeMemory(
        List<float> vectorObs,
        List<ISensor> sensors,
        float reward,
        bool done,
        List<float> memory)
    {
        return new List<float>();
    }
}
