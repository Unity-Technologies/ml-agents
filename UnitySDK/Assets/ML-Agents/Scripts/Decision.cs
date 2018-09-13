using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Interface for implementing the behavior of an Agent that uses a Heuristic
    /// Brain. The behavior of an Agent in this case is fully decided using the
    /// implementation of these methods and no training or inference takes place.
    /// Currently, the Heuristic Brain does not support text observations and actions.
    /// </summary>
    public abstract class Decision : ScriptableObject
    {
        public BrainParameters brainParameters;
        
        /// <summary>
        /// Defines the decision-making logic of the agent. Given the information 
        /// about the agent, returns a vector of actions.
        /// </summary>
        /// <returns>Vector action vector.</returns>
        /// <param name="vectorObs">The vector observations of the agent.</param>
        /// <param name="visualObs">The cameras the agent uses for visual observations.</param>
        /// <param name="reward">The reward the agent received at the previous step.</param>
        /// <param name="done">Whether or not the agent is done.</param>
        /// <param name="memory">
        /// The memories stored from the previous step with 
        /// <see cref="MakeMemory(List{float}, List{Texture2D}, float, bool, List{float})"/>
        /// </param>
        public abstract float[] Decide(
            List<float>
                vectorObs,
            List<Texture2D> visualObs,
            float reward,
            bool done,
            List<float> memory);

        /// <summary>
        /// Defines the logic for creating the memory vector for the Agent.
        /// </summary>
        /// <returns>The vector of memories the agent will use at the next step.</returns>
        /// <param name="vectorObs">The vector observations of the agent.</param>
        /// <param name="visualObs">The cameras the agent uses for visual observations.</param>
        /// <param name="reward">The reward the agent received at the previous step.</param>
        /// <param name="done">Whether or not the agent is done.</param>
        /// <param name="memory">
        /// The memories stored from the previous call to this method.
        /// </param>
        public abstract List<float> MakeMemory(
            List<float> vectorObs,
            List<Texture2D> visualObs,
            float reward,
            bool done,
            List<float> memory);
    }
}
