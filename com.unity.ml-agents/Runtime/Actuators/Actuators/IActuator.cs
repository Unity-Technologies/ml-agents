using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Policies;
using UnityEngine;

namespace Unity.MLAgents.Actuators
{
    public readonly struct ActuatorSpace
    {
        public SpaceType ActuatorSpaceType =>
            ContinuousActionSpaceDef.NumActions > 0 ? SpaceType.Continuous : SpaceType.Discrete;
        public ActionSpaceDef ContinuousActionSpaceDef { get; }
        public ActionSpaceDef DiscreteActionSpaceDef { get; }

        public ActuatorSpace(ActionSpaceDef continuousActionSpaceDef,
            ActionSpaceDef discreteActionSpaceDef)
        {
            ContinuousActionSpaceDef = continuousActionSpaceDef;
            DiscreteActionSpaceDef = discreteActionSpaceDef;
        }

    }
    /// <summary>
    /// Abstraction that facilitates the execution of actions.
    /// </summary>
    public interface IActuator : IActionReceiver
    {
        int TotalNumberOfActions { get; }

        ActuatorSpace ActuatorSpace { get; }

        /// <summary>
        /// Gets the name of this IActuator which will be used to sort it.
        /// </summary>
        /// <returns></returns>
        string GetName();

        void ResetData();
    }
}
