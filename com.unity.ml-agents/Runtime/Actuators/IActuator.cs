using System;
using UnityEngine;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// Abstraction that facilitates the execution of actions.
    /// </summary>
    internal interface IActuator : IActionReceiver
    {
        int TotalNumberOfActions { get; }

        ActionSpaceDef ActionSpaceDef { get; }

        /// <summary>
        /// Gets the name of this IActuator which will be used to sort it.
        /// </summary>
        /// <returns></returns>
        string Name { get; }

        void ResetData();
    }
}
