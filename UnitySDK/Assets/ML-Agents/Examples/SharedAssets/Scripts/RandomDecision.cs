using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    public class RandomDecision : Decision
    {
        public override float[] Decide(
            List<float> vectorObs,
            List<Texture2D> visualObs,
            float reward,
            bool done,
            List<float> memory)
        {
            if (brainParameters.vectorActionSpaceType == SpaceType.Continuous)
            {
                var act = new List<float>();

                for (var i = 0; i < brainParameters.vectorActionSize[0]; i++)
                {
                    act.Add(2 * Random.value - 1);
                }

                return act.ToArray();
            }
            else
            {
                var act = new float[brainParameters.vectorActionSize.Length];
                for (var i = 0; i < brainParameters.vectorActionSize.Length; i++)
                {
                    act[i] = Random.Range(0, brainParameters.vectorActionSize[i]);
                }
                return act;
            }
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
