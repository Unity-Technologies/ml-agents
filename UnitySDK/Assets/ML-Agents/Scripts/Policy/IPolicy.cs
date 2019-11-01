using System;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// IPolicy is connected to a single Agent. Each time the agent needs
    /// a decision, it will request a decision to the Policy. The decision
    /// will not be taken immediately but will be taken before or when 
    /// DecideAction is called.
    /// </summary>
    public interface IPolicy : IDisposable
    {
        /// <summary>
        /// Signals the Brain that the Agent needs a Decision. The Policy
        /// will make the decision at a later time to allow possible
        /// batching of requests.
        /// </summary>
        /// <param name="agent"></param>
        void RequestDecision(Agent agent);

        /// <summary>
        /// Signals the Policy that if the Decision has not been taken yet,
        /// it must be taken now. The Brain is expected to update the actions
        /// of the Agents at this point the latest.
        /// </summary>
        void DecideAction();
    }
}
