using Barracuda;
using System;


namespace MLAgents
{
    public static class BrainFactory
    {
        public static IBrain GenerateBrain(
            BrainParameters brainParameters,
            string behaviorName,
            NNModel model,
            bool useHeuristic,
            Func<float[]> heuristic,
            InferenceDevice inferenceDevice = InferenceDevice.CPU
        )
        {
            if (model == null || useHeuristic)
            {
                return new HeuristicBrain(heuristic);

            }
            else
            {
                return new LearningBrain(
                    brainParameters, model, inferenceDevice, behaviorName);
            }
        }
    }
}
