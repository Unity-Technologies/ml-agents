using System;
<<<<<<< HEAD
||||||| constructed merge base
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Policies;
=======
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Policies;
>>>>>>> Get discrete action mask working and backward compatible.
using UnityEngine;

namespace Unity.MLAgents.Actuators
{
<<<<<<< HEAD
||||||| constructed merge base
    public struct ActuatorSpace
    {
        public readonly SpaceType[] SpaceTypes;
        public readonly int[] BranchSizes;
        public readonly int NumActions;

        public static ActuatorSpace MakeContinuous(int numActions)
        {
            var spaceTypes = new SpaceType[numActions];
            for (var i = 0; i < numActions; i++)
            {
                spaceTypes[i] = SpaceType.Continuous;
            }
            var actuatorSpace = new ActuatorSpace(spaceTypes, numActions);
            return actuatorSpace;
        }

        public static ActuatorSpace MakeDiscrete(int[] branchSizes)
        {
            var numActions = branchSizes.Length;
            var spaceTypes = new SpaceType[numActions];
            for (var i = 0; i < numActions; i++)
            {
                spaceTypes[i] = SpaceType.Discrete;
            }
            var actuatorSpace = new ActuatorSpace(spaceTypes, numActions, branchSizes);
            return actuatorSpace;
        }

        ActuatorSpace(SpaceType[] spaceTypes, int numActions, int[] branchSizes = null)
        {
            SpaceTypes = spaceTypes;
            NumActions = numActions;
            BranchSizes = branchSizes;
        }
    }

=======
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
>>>>>>> Get discrete action mask working and backward compatible.
    /// <summary>
    /// Abstraction that facilitates the execution of actions.
    /// </summary>
    public interface IActuator : IActionReceiver
    {
<<<<<<< HEAD
        int TotalNumberOfActions { get; }
||||||| constructed merge base
        ActionSegment<int> DiscreteActions
        {
            get;
        }

        ActionSegment<float> ContinuousActions
        {
            get;
        }

        int TotalNumberOfActions
        {
            get;
        }

        ActuatorSpace DiscreteActuatorSpace
        {
            get;
        }

        ActuatorSpace ContinuousActuatorSpace
        {
            get;
        }
=======
        int TotalNumberOfActions { get; }

        ActuatorSpace ActuatorSpace { get; }
>>>>>>> Get discrete action mask working and backward compatible.

        /// <summary>
        /// Gets the name of this IActuator which will be used to sort it.
        /// </summary>
        /// <returns></returns>
        string Name { get; }

        void ResetData();
    }
}
