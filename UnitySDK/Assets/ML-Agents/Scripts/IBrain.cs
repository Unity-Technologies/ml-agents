using System;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Brain receive data from Agents through calls to SubscribeAgentForDecision. The brain then updates the
    /// actions of the agents at each FixedUpdate.
    /// The Brain encapsulates the decision making process. Every Agent must be assigned a Brain,
    /// but you can use the same Brain with more than one Agent. You can also create several
    /// Brains, attach each of the Brain to one or more than one Agent.
    /// Brain assets has several important properties that you can set using the Inspector window.
    /// These properties must be appropriate for the Agents using the Brain. For example, the
    /// Vector Observation Space Size property must match the length of the feature
    /// vector created by an Agent exactly.
    /// </summary>
    public interface IBrain : IDisposable
    {
        void RequestDecision(Agent agent);
        void DecideAction();
    }
}
