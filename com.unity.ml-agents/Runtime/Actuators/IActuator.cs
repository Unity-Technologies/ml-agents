using System;
using UnityEngine;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// Abstraction that facilitates the execution of actions.
    /// </summary>
    public interface IActuator : IActionReceiver
    {
        /// <summary>
        /// Gets the name of this IActuator which will be used to sort it.
        /// </summary>
        /// <returns></returns>
        string Name { get; }

        /// <summary>
        /// Resets the internal state of the actuator. This is called at the end of an Agent's episode.
        /// Most implementations can leave this empty.
        /// </summary>
        void ResetData();
    }
}
