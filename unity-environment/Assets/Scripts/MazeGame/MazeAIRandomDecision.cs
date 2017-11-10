using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MazeAIRandomDecision : MonoBehaviour, Decision {

	public float[] Decide (List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
	{
        int result = Random.Range(0, 4);
		return new float[]{ result};

	}

	public float[] MakeMemory (List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
	{
		return default(float[]);
		
	}
}
