using Barracuda;
using System;


namespace MLAgents
{
    /// <summary>
    /// The Factory to generate policies. 
    /// </summary>
    public static class PolicyFactory
    {
        /// <summary>
        /// This method will generate a new IPolicy depending on
        /// the parameters fed as input. 
        /// </summary>
        /// <param name="brainParameters"> 
        /// The parameters of the Policy. 
        /// Example : Vector Observation size 
        /// </param>
        /// <param name="useHeuristicPolicy">
        /// If true, the Policy generated will be a HeuristicPolicy.
        /// </param>
        /// <param name="heuristic">
        /// The Heuristic method used by the HeuristicPolicy
        /// </param>
        /// <param name="useRemotePolicy">
        /// If true, (and useHeuristicPolicy is false)
        ///  the Policy generated will be a RemotePolicy
        /// </param>
        /// <param name="behaviorName">
        /// The name of the Behavior used by the RemotePolicy
        /// </param>
        /// <param name="model">
        /// The Barracuda Model used for the BarracudaPolicy
        /// </param>

        /// <param name="inferenceDevice">
        /// The inference device used by the Barracuda Policy</param>
        /// <returns></returns>
        public static IPolicy GeneratePolicy(
            BrainParameters brainParameters,
            bool useHeuristicPolicy,
            Func<float[]> heuristic,
            bool useRemotePolicy,
            string behaviorName,
            NNModel model,
            InferenceDevice inferenceDevice = InferenceDevice.CPU
        )
        {
            if (model == null || useHeuristicPolicy)
            {
                return new HeuristicPolicy(heuristic);
            }
            if (useRemotePolicy)
            {
                return new RemotePolicy(brainParameters, behaviorName);
            }
            else
            {
                return new BarracudaPolicy(brainParameters, model, inferenceDevice);
            }
        }
    }
}
