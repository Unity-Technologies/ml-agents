using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomDecision : MonoBehaviour, Decision {



    public float[] Decide(List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
    {
        if (gameObject.GetComponent<Brain>().brainParameters.actionSpaceType == StateType.continuous)
        {
            int actionSize = gameObject.GetComponent<Brain>().brainParameters.actionSize;
            float[] result = new float[actionSize];
            for (int i = 0; i < actionSize; i++)
            {
                result[i] = Random.value * 2 - 1;
            }
            return result;
        }
        else
        {
            int actionSize = gameObject.GetComponent<Brain>().brainParameters.actionSize;
            float[] result = new float[1];
            result[0] = (float)Random.Range(0, actionSize);
            return result;
        }

    }

    public float[] MakeMemory(List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
    {
        return new float[0];

    }
}
