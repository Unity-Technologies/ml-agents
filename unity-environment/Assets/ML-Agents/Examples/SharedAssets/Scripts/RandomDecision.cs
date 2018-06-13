using System.Collections.Generic;
using UnityEngine;
using MLAgents;

namespace MLAgents
{

    public class RandomDecision :Decision
    {

        public override float[] Decide(
            List<float> vectorObs,
            List<Texture2D> visualObs,
            float reward,
            bool done,
            List<float> memory)
        {
            if (brainParameters.vectorActionSpaceType == SpaceType.continuous)
            {
                List<float> act = new List<float>();

                for (int i = 0; i < brainParameters.vectorActionSize; i++)
                {
                    act.Add(2 * Random.value - 1);
                }

                return act.ToArray();
            }

            return new float[1] {Random.Range(0, brainParameters.vectorActionSize)};
        }

        public override List<float> MakeMemory(
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
