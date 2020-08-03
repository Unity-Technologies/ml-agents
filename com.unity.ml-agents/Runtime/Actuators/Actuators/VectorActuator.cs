using System;
using Unity.MLAgents.Policies;

namespace Unity.MLAgents.Actuators
{
    class VectorActuator : IActuator
    {
        // Easy access for now about which space type to use for business logic.
        // Should be removed once a mixed SpaceType NN is available.
        string m_Name;
        IActionReceiver m_ActionReceiver;

        ActionBuffers m_ActionBuffers;
        ActuatorSpace m_ActuatorSpace;

        public ActuatorSpace ActuatorSpace
        {
            get => m_ActuatorSpace;
            private set => m_ActuatorSpace = value;
        }

        public VectorActuator(IActionReceiver actionReceiver,
            int[] vectorActionSize,
            SpaceType spaceType,
            string name = "VectorActuator")
        {
            m_ActionReceiver = actionReceiver;
            ActionSpaceType = spaceType;
            ActuatorSpace = new ActuatorSpace();
            string suffix;
            ActionSpaceDef discreteActionSpace, continuousActionSpace;
            switch (ActionSpaceType)
            {
                case SpaceType.Continuous:
                    continuousActionSpace = ActionSpaceDef.MakeContinuous(vectorActionSize[0]);
                    discreteActionSpace = ActionSpaceDef.MakeDiscrete(Array.Empty<int>());
                    suffix = "-Continuous";
                    break;
                case SpaceType.Discrete:
                    discreteActionSpace = ActionSpaceDef.MakeDiscrete(vectorActionSize);
                    continuousActionSpace = ActionSpaceDef.MakeContinuous(0);
                    suffix = "-Discrete";
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(spaceType),
                        spaceType,
                        "Unknown enum value.");
            }

            m_Name = name + suffix;
            ActuatorSpace = new ActuatorSpace(continuousActionSpace, discreteActionSpace);
        }

        public SpaceType ActionSpaceType { get; }

        public void ResetData()
        {
            m_ActionBuffers.DiscreteActions = new ActionSegment<int>();
            m_ActionBuffers.ContinuousActions = new ActionSegment<float>();
        }

        public void OnActionReceived(ActionBuffers actionBuffers)
        {
            m_ActionBuffers = actionBuffers;
            m_ActionReceiver.OnActionReceived(m_ActionBuffers);
        }

        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            if (ActionSpaceType == SpaceType.Discrete)
            {
                m_ActionReceiver.WriteDiscreteActionMask(actionMask);
            }
        }


        public int TotalNumberOfActions => m_ActuatorSpace.ContinuousActionSpaceDef.NumActions +
            m_ActuatorSpace.DiscreteActionSpaceDef.NumActions;

        public string GetName()
        {
            return m_Name;
        }
    }
}
