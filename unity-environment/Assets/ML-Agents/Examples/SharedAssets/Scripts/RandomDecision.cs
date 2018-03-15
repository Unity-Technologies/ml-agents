using System.Collections.Generic;
using UnityEngine;

public class RandomDecision : MonoBehaviour, Decision
{
    BrainParameters brainParameters;
    SpaceType actionSpaceType;
    int actionSpaceSize;

    public void Awake()
    {
        brainParameters =
            gameObject.GetComponent<Brain>().brainParameters;
        actionSpaceType = brainParameters.vectorActionSpaceType;
        actionSpaceSize = brainParameters.vectorActionSize;
    }

    public float[] Decide(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {
        if (actionSpaceType == SpaceType.continuous)
        {
            List<float> act = new List<float>();

            for (int i = 0; i < actionSpaceSize; i++)
            {
                act.Add(2 * Random.value - 1);
            }

            return act.ToArray();
        }

        return new float[1] { Random.Range(0, actionSpaceSize) };
    }

    public List<float> MakeMemory(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {
        return new List<float>();
    }
}
