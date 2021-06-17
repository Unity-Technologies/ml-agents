//Put this script on your blue cube.

using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class ParameterizedAgent : Agent
{
    public RewardType m_RewardType = RewardType.Time;

    public int stepvalue=5000;
    
    public void setMaxStep(int value)
    {
        stepvalue = value;
        MaxStep = value;
    }
}
