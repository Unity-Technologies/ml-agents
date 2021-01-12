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
        /// <param name="actionSpec"></param>
        /// <param name="name"></param>
        public VectorActuator(IActionReceiver actionReceiver,
                              ActionSpec actionSpec,
                              string name = "VectorActuator")
        {
            m_ActionReceiver = actionReceiver;
            ActionSpec = actionSpec;
            string suffix;
            if (actionSpec.NumContinuousActions == 0)
            {
                suffix = "-Discrete";
            }
            else if (actionSpec.NumDiscreteActions == 0)
            {
                suffix = "-Continuous";
            }
            else
            {
                suffix = $"-Continuous-{actionSpec.NumContinuousActions}-Discrete-{actionSpec.NumDiscreteActions}";
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
