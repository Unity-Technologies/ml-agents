using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BasicDecision : MonoBehaviour, Decision
{

    public float[] Decide(List<float> state, List<Texture2D> observation, float reward, bool done, List<float> memory)
    {
        return new float[1]{ 1f };

    }

    public List<float> MakeMemory(List<float> state, List<Texture2D> observation, float reward, bool done, List<float> memory)
    {
        return new List<float>();

    }
}
