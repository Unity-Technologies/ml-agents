using Barracuda;
using System;


namespace MLAgents
{
    public static class BrainFactory
    {
        public static IBrain GenerateBrain(
            BrainParameters brainParameters,
            bool useRemoteBrain,
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
            if (useRemoteBrain)
            {
                return new RemoteBrain(brainParameters, behaviorName);
            }
            else
            {
                return new BarracudaBrain(brainParameters, model, inferenceDevice);
            }
        }
    }
}
