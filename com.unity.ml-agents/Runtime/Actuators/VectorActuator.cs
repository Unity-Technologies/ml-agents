using System;

using Unity.MLAgents.Policies;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// IActuator implementation that forwards to an <see cref="IActionReceiver"/>.
    /// </summary>
    internal class VectorActuator : IActuator
    {
        IActionReceiver m_ActionReceiver;

        ActionBuffers m_ActionBuffers;
        internal ActionBuffers ActionBuffers
        {
            get => m_ActionBuffers;
            private set => m_ActionBuffers = value;
        }

        /// <summary>
        /// Create a VectorActuator that forwards to the provided IActionReceiver.
        /// </summary>
        /// <param name="actionReceiver">The <see cref="IActionReceiver"/> used for OnActionReceived and WriteDiscreteActionMask.</param>
        /// <param name="vectorActionSize">For discrete action spaces, the branch sizes for each action.
        /// For continuous action spaces, the number of actions is the 0th element.</param>
        /// <param name="spaceType"></param>
        /// <param name="name"></param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for invalid <see cref="SpaceType"/></exception>
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
                    ActionSpec = ActionSpec.MakeContinuous(vectorActionSize[0]);
                    suffix = "-Continuous";
                    break;
                case SpaceType.Discrete:
                    ActionSpec = ActionSpec.MakeDiscrete(vectorActionSize);
                    suffix = "-Discrete";
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(spaceType),
                        spaceType,
                        "Unknown enum value.");
            }
            Name = name + suffix;
        }

        /// <inheritdoc />
        public void ResetData()
        {
            m_ActionBuffers = ActionBuffers.Empty;
        }

        /// <inheritdoc />
        public void OnActionReceived(ActionBuffers actionBuffers)
        {
            ActionBuffers = actionBuffers;
            m_ActionReceiver.OnActionReceived(ActionBuffers);
        }

        /// <inheritdoc />
        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            m_ActionReceiver.WriteDiscreteActionMask(actionMask);
        }

        /// <inheritdoc/>
        public ActionSpec ActionSpec { get; }

        /// <inheritdoc />
        public string Name { get; }
    }
}
