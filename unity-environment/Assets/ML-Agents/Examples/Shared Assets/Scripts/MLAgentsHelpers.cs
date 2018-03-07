using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class MLAgentsHelpers
{
    
    /// <summary>
    /// Add the x,y,z of the rotation to the state list.
    /// </summary>
    public static void CollectRotationState(Agent agent, Transform t)
    {
        agent.AddVectorObs(t.rotation.eulerAngles.x / 180.0f - 1.0f);
        agent.AddVectorObs(t.rotation.eulerAngles.y / 180.0f - 1.0f);
        agent.AddVectorObs(t.rotation.eulerAngles.z / 180.0f - 1.0f);
    }
}
