using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Ball3DDecision : MonoBehaviour, Decision
{
    public float[] Decide(List<float> state, List<Texture2D> observation, float reward, bool done, float[] memory)
    {
        if (gameObject.GetComponent<Brain>().brainParameters.actionSpaceType == StateType.continuous)
        {
            return new float[2]{ 0f, 0f};

        }
        else
        {
            return new float[1]{ 1f };
        }
    }

    public float[] MakeMemory(List<float> state, List<Texture2D> observation, float reward, bool done, float[] memory)
    {
        return new float[0];
    }
}
