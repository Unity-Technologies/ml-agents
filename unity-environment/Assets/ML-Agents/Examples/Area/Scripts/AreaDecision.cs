using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AreaDecision : MonoBehaviour, Decision {

    public float[] Decide (List<float> state, List<Texture2D> observation, float reward, bool done, List<float> memory)
    {
        float[] action = new float[1];
        action[0] = Random.Range(0, 5);
        return action;
    }

    public List<float> MakeMemory (List<float> state, List<Texture2D> observation, float reward, bool done, List<float> memory)
    {
        return new List<float>();
        
    }
}
