using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BasicDecision : MonoBehaviour, Decision {

	public float[] Decide (List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
	{
		return default(float[]);

	}

	public float[] MakeMemory (List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
	{
		return default(float[]);
		
	}
}
