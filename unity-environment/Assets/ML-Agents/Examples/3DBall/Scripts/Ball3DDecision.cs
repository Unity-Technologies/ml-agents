using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Ball3DDecision : MonoBehaviour, Decision
{
    public float[] Decide(List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
    {
        if (gameObject.GetComponent<Brain>().brainParameters.actionSpaceType == StateType.continuous)
        {
            List<float> ret = new List<float>();
            if (state[2] < 0 || state[5] < 0)
            {
                ret.Add(state[5]);
            }
            else
            {
                ret.Add(state[5]);
            }
            if (state[3] < 0 || state[7] < 0)
            {
                ret.Add(-state[7]);
            }
            else
            {
                ret.Add(-state[7]);
            }
            return ret.ToArray();

        }
        else
        {
            return new float[1]{ 1f };
        }
    }

    public float[] MakeMemory(List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
    {
        return new float[0];
    }
}
