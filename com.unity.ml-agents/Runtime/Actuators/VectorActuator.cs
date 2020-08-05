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
        internal ActionBuffers ActionBuffers
        {
            get => m_ActionBuffers;
            private set => m_ActionBuffers = value;
        }

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
            m_ActionBuffers = ActionBuffers.Empty;
        }

        public void OnActionReceived(ActionBuffers actionBuffers)
        {
            ActionBuffers = actionBuffers;
            m_ActionReceiver.OnActionReceived(ActionBuffers);
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
        public int TotalNumberOfActions => ActionSpaceDef.NumContinuousActions +
        ActionSpaceDef.NumDiscreteActions;

        /// <summary>
        /// <inheritdoc cref="IActuator.ActionSpaceDef"/>
        /// </summary>
        public ActionSpaceDef ActionSpaceDef { get; }

        public string Name { get; }
    }
}
