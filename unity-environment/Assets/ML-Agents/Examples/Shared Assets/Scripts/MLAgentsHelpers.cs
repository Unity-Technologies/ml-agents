using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class MLAgentsHelpers{

    /// <summary>
    /// Add the x,y,z of this Vector3 to the state list.
    /// </summary>
    public static void CollectVector3State(Agent agent, Vector3 v)
    {
        agent.AddVectorObs(v.x);
        agent.AddVectorObs(v.y);
        agent.AddVectorObs(v.z);
    }

    /// <summary>
    /// Add the x,y,z of the rotation to the state list.
    /// </summary>
    public static void CollectRotationState(Agent agent, Transform t)
    {
        agent.AddVectorObs(t.rotation.eulerAngles.x / 180.0f - 1.0f);
        agent.AddVectorObs(t.rotation.eulerAngles.y / 180.0f - 1.0f);
        agent.AddVectorObs(t.rotation.eulerAngles.z / 180.0f - 1.0f);
    }

    /// <summary>
    /// We can only collect floats in CollecState,
    /// so in some cases it is helpful to convert bools to floats.
    /// </summary>
    public static float ConvertBoolToFloat(bool b)
    {
        float f = b == true? 1 : 0;
        return f;
    }
	
}
