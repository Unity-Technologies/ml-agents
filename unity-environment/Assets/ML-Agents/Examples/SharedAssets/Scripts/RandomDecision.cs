using System.Collections.Generic;
using UnityEngine;
using MLAgents;

namespace MLAgents
{

    public class RandomDecision : MonoBehaviour, Decision
    {
        BrainParameters brainParameters;
        SpaceType actionSpaceType;

        public void Awake()
        {
            brainParameters =
                gameObject.GetComponent<Brain>().brainParameters;
            actionSpaceType = brainParameters.vectorActionSpaceType;
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

                for (int i = 0; i < brainParameters.vectorActionSize[0]; i++)
                {
                    act.Add(2 * Random.value - 1);
                }

                return act.ToArray();
            }
            else
            {
                float[] act = new float[brainParameters.vectorActionSize.Length];
                for (int i = 0; i < brainParameters.vectorActionSize.Length; i++)
                {
                    act[i]=Random.Range(0, brainParameters.vectorActionSize[i]);
                }
                return act;
            }
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
}
