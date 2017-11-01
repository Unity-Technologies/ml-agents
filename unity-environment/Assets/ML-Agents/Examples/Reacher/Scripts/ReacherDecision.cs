using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ReacherDecision : MonoBehaviour, Decision {

	public float[] Decide (List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
	{
        float[] action = new float[4];
        for (int i = 0; i < 4; i++) {
            action[i] = Random.Range(-1f, 1f);
        }
        return action;

	}

	public float[] MakeMemory (List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
	{
		return default(float[]);
		
	}
}
