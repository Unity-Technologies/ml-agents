using System;

using Unity.MLAgents.Policies;

namespace Unity.MLAgents.Actuators
{
    public class VectorActuator : IActuator
    {
<<<<<<< HEAD
||||||| constructed merge base
        // Easy access for now about which space type to use for business logic.
        // Should be removed once a mixed SpaceType NN is available.
        SpaceType m_SpaceType;
        string m_Name;
=======
        // Easy access for now about which space type to use for business logic.
        // Should be removed once a mixed SpaceType NN is available.
        string m_Name;
>>>>>>> Get discrete action mask working and backward compatible.
        IActionReceiver m_ActionReceiver;

<<<<<<< HEAD
        ActionBuffers m_ActionBuffers;
        internal ActionBuffers ActionBuffers
        {
            get => m_ActionBuffers;
            private set => m_ActionBuffers = value;
        }

||||||| constructed merge base
=======
        ActionBuffers m_ActionBuffers;
        ActuatorSpace m_ActuatorSpace;

        public ActuatorSpace ActuatorSpace
        {
            get => m_ActuatorSpace;
            private set => m_ActuatorSpace = value;
        }

>>>>>>> Get discrete action mask working and backward compatible.
        public VectorActuator(IActionReceiver actionReceiver,
            int[] vectorActionSize,
            SpaceType spaceType,
            string name = "VectorActuator")
        {
            m_ActionReceiver = actionReceiver;
<<<<<<< HEAD
||||||| constructed merge base
            m_SpaceType = spaceType;
=======
            ActionSpaceType = spaceType;
            ActuatorSpace = new ActuatorSpace();
>>>>>>> Get discrete action mask working and backward compatible.
            string suffix;
<<<<<<< HEAD
            switch (spaceType)
||||||| constructed merge base
            switch (m_SpaceType)
=======
            ActionSpaceDef discreteActionSpace, continuousActionSpace;
            switch (ActionSpaceType)
>>>>>>> Get discrete action mask working and backward compatible.
            {
                case SpaceType.Continuous:
<<<<<<< HEAD
                    ActionSpec = ActionSpec.MakeContinuous(vectorActionSize[0]);
||||||| constructed merge base
                    ContinuousActuatorSpace = ActuatorSpace.MakeContinuous(vectorActionSize[0]);
                    DiscreteActuatorSpace = ActuatorSpace.MakeDiscrete(Array.Empty<int>());
=======
                    continuousActionSpace = ActionSpaceDef.MakeContinuous(vectorActionSize[0]);
                    discreteActionSpace = ActionSpaceDef.MakeDiscrete(Array.Empty<int>());
>>>>>>> Get discrete action mask working and backward compatible.
                    suffix = "-Continuous";
                    break;
                case SpaceType.Discrete:
<<<<<<< HEAD
                    ActionSpec = ActionSpec.MakeDiscrete(vectorActionSize);
||||||| constructed merge base
                    DiscreteActuatorSpace = ActuatorSpace.MakeDiscrete(vectorActionSize);
                    ContinuousActuatorSpace = ActuatorSpace.MakeContinuous(0);
=======
                    discreteActionSpace = ActionSpaceDef.MakeDiscrete(vectorActionSize);
                    continuousActionSpace = ActionSpaceDef.MakeContinuous(0);
>>>>>>> Get discrete action mask working and backward compatible.
                    suffix = "-Discrete";
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(spaceType),
                        spaceType,
                        "Unknown enum value.");
            }
<<<<<<< HEAD
            Name = name + suffix;
||||||| constructed merge base
            m_Name = name + suffix;
=======

            m_Name = name + suffix;
            ActuatorSpace = new ActuatorSpace(continuousActionSpace, discreteActionSpace);
>>>>>>> Get discrete action mask working and backward compatible.
        }

        public SpaceType ActionSpaceType { get; }

        public void ResetData()
        {
<<<<<<< HEAD
            m_ActionBuffers = ActionBuffers.Empty;
||||||| constructed merge base
            DiscreteActions = new ActionSegment<int>();
            ContinuousActions = new ActionSegment<float>();
=======
            m_ActionBuffers.DiscreteActions = new ActionSegment<int>();
            m_ActionBuffers.ContinuousActions = new ActionSegment<float>();
>>>>>>> Get discrete action mask working and backward compatible.
        }

        public void OnActionReceived(ActionBuffers actionBuffers)
        {
<<<<<<< HEAD
            ActionBuffers = actionBuffers;
            m_ActionReceiver.OnActionReceived(ActionBuffers);
||||||| constructed merge base
            ContinuousActions = continuousActions;
            DiscreteActions = discreteActions;
            m_ActionReceiver.OnActionReceived(ContinuousActions, DiscreteActions);
=======
            m_ActionBuffers = actionBuffers;
            m_ActionReceiver.OnActionReceived(m_ActionBuffers);
>>>>>>> Get discrete action mask working and backward compatible.
        }

        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
<<<<<<< HEAD
                m_ActionReceiver.WriteDiscreteActionMask(actionMask);
||||||| constructed merge base
            if (m_SpaceType == SpaceType.Discrete)
            {
                // TODO: Call into agent?
            }
=======
            if (ActionSpaceType == SpaceType.Discrete)
            {
                m_ActionReceiver.WriteDiscreteActionMask(actionMask);
            }
>>>>>>> Get discrete action mask working and backward compatible.
        }

<<<<<<< HEAD
        /// <summary>
        /// Returns the number of discrete branches + the number of continuous actions.
        /// </summary>
        public int TotalNumberOfActions => ActionSpec.NumContinuousActions +
        ActionSpec.NumDiscreteActions;
||||||| constructed merge base
        public ActionSegment<int> DiscreteActions { get; private set; }
        public ActionSegment<float> ContinuousActions { get; private set; }
        public int TotalNumberOfActions
        {
            get { return ContinuousActuatorSpace.NumActions + DiscreteActuatorSpace.NumActions; }
        }
        public ActuatorSpace DiscreteActuatorSpace { get; }
        public ActuatorSpace ContinuousActuatorSpace { get; }
=======

        public int TotalNumberOfActions => m_ActuatorSpace.ContinuousActionSpaceDef.NumActions +
            m_ActuatorSpace.DiscreteActionSpaceDef.NumActions;
>>>>>>> Get discrete action mask working and backward compatible.

        /// <summary>
        /// <inheritdoc cref="IActionReceiver.ActionSpec"/>
        /// </summary>
        public ActionSpec ActionSpec { get; }

        public string Name { get; }
    }
}
