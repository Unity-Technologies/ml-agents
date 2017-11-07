using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleRobotDecision : MonoBehaviour, Decision {

    public float[] Decide(List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
    {
        if (gameObject.GetComponent<Brain>().brainParameters.actionSpaceType == StateType.continuous)
        {
            Debug.LogWarning("Simple robot is not designed for conttinuous space");
            return new float[4] { 0f, 0f, 0f, 0.0f };
        }
        else
        {
            return new float[1] { 0f };
        }
    }

    public float[] MakeMemory(List<float> state, List<Camera> observation, float reward, bool done, float[] memory)
    {
        return new float[0];
    }

}
