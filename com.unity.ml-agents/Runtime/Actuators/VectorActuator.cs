using System;
using Unity.MLAgents.Policies;

namespace Unity.MLAgents.Actuators
{
    internal class VectorActuator : IActuator
    {
        // Easy access for now about which space type to use for business logic.
        // Should be removed once a mixed SpaceType NN is available.
        IActionReceiver m_ActionReceiver;

        ActionBuffers m_ActionBuffers;
        ActionSpaceDef m_ActionSpaceDef;

        public VectorActuator(IActionReceiver actionReceiver,
            int[] vectorActionSize,
            SpaceType spaceType,
            string name = "VectorActuator")
        {
            m_ActionReceiver = actionReceiver;
            string suffix;
            switch (spaceType)
            {
                case SpaceType.Continuous:
                    ActionSpaceDef = ActionSpaceDef.MakeContinuous(vectorActionSize[0]);
                    suffix = "-Continuous";
                    break;
                case SpaceType.Discrete:
                    ActionSpaceDef = ActionSpaceDef.MakeDiscrete(vectorActionSize);
                    suffix = "-Discrete";
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(spaceType),
                        spaceType,
                        "Unknown enum value.");
            }
            Name = name + suffix;
        }

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
            if (ActionSpaceDef.SpaceType == SpaceType.Discrete)
            {
                m_ActionReceiver.WriteDiscreteActionMask(actionMask);
            }
        }

        /// <summary>
        /// Returns the number of discrete branches + the number of continuous actions.
        /// </summary>
        public int TotalNumberOfActions => m_ActionSpaceDef.NumContinuousActions +
            m_ActionSpaceDef.NumDiscreteActions;

        /// <summary>
        /// <inheritdoc cref="IActuator.ActionSpaceDef"/>
        /// </summary>
        public ActionSpaceDef ActionSpaceDef
        {
            get => m_ActionSpaceDef;
            private set => m_ActionSpaceDef = value;
        }

        public string Name { get; }

    }
}
