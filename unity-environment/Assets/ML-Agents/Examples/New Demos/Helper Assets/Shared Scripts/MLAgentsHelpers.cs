using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class MLAgentsHelpers{

    //Add the x,y,z of this Vector3 to the state list
    public static void CollectVector3State(List<float> state, Vector3 v)
    {
        state.Add(v.x);
        state.Add(v.y);
        state.Add(v.z);
    }

	//add the x,y,z of the rotation to the state
    public static void CollectRotationState(List<float> state, Transform t)
    {
		state.Add(t.rotation.eulerAngles.x/180.0f-1.0f);
		state.Add(t.rotation.eulerAngles.y/180.0f-1.0f);
		state.Add(t.rotation.eulerAngles.z/180.0f-1.0f);
    }
    public static void CollectLocalRotationState(List<float> state, Transform t)
    {
		state.Add(t.localRotation.eulerAngles.x/180.0f-1.0f);
		state.Add(t.localRotation.eulerAngles.y/180.0f-1.0f);
		state.Add(t.localRotation.eulerAngles.z/180.0f-1.0f);
    }

    //we can only collect floats in CollecState so we need to convert bools to floats
    public static float ConvertBoolToFloat(bool b)
    {
        float f = b == true? 1 : 0;
        return f;
    }
	
}
