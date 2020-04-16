using System;
using System.Collections.Generic;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Policies
{
    /// <summary>
    /// IPolicy is connected to a single Agent. Each time the agent needs
    /// a decision, it will request a decision to the Policy. The decision
    /// will not be taken immediately but will be taken before or when
    /// DecideAction is called.
    /// </summary>
    internal interface IPolicy : IDisposable
    {
        /// <summary>
        /// Signals the Brain that the Agent needs a Decision. The Policy
        /// will make the decision at a later time to allow possible
        /// batching of requests.
        /// </summary>
        /// <param name="info"></param>
        /// <param name="sensors"></param>
        void RequestDecision(AgentInfo info, List<ISensor> sensors);

        /// <summary>
        /// Signals the Policy that if the Decision has not been taken yet,
        /// it must be taken now. The Brain is expected to update the actions
        /// of the Agents at this point the latest.
        /// </summary>
        float[] DecideAction();
    }
}
