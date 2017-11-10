using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PongAIDecision : MonoBehaviour, Decision {

	public float[] Decide (List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
	{
        int result = 1;
        if(state[3] > state[0])
        {
            result = 2;
        }
        else
        {
            result = 0;
        }
		return new float[]{ result};

	}

	public float[] MakeMemory (List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
	{
		return default(float[]);
		
	}
}
