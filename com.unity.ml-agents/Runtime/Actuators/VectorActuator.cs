using UnityEngine.Profiling;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// IActuator implementation that forwards calls to an <see cref="IActionReceiver"/> and an <see cref="IHeuristicProvider"/>.
    /// </summary>
    internal class VectorActuator : IActuator, IBuiltInActuator
    {
        IActionReceiver m_ActionReceiver;
        IHeuristicProvider m_HeuristicProvider;

        ActionBuffers m_ActionBuffers;
        internal ActionBuffers ActionBuffers
        {
            get => m_ActionBuffers;
            private set => m_ActionBuffers = value;
        }

        /// <summary>
        /// Create a VectorActuator that forwards to the provided IActionReceiver.
        /// </summary>
        /// <param name="actionReceiver">The <see cref="IActionReceiver"/> used for OnActionReceived and WriteDiscreteActionMask.
        /// If this parameter also implements <see cref="IHeuristicProvider"/> it will be cast and used to forward calls to
        /// <see cref="IHeuristicProvider.Heuristic"/>.</param>
        /// <param name="actionSpec"></param>
        /// <param name="name"></param>
        public VectorActuator(IActionReceiver actionReceiver,
                              ActionSpec actionSpec,
                              string name = "VectorActuator")
            : this(actionReceiver, actionReceiver as IHeuristicProvider, actionSpec, name) { }

        /// <summary>
        /// Create a VectorActuator that forwards to the provided IActionReceiver.
        /// </summary>
        /// <param name="actionReceiver">The <see cref="IActionReceiver"/> used for OnActionReceived and WriteDiscreteActionMask.</param>
        /// <param name="heuristicProvider">The <see cref="IHeuristicProvider"/> used to fill the <see cref="ActionBuffers"/>
        /// for Heuristic Policies.</param>
        /// <param name="actionSpec"></param>
        /// <param name="name"></param>
        public VectorActuator(IActionReceiver actionReceiver,
                              IHeuristicProvider heuristicProvider,
                              ActionSpec actionSpec,
                              string name = "VectorActuator")
        {
            m_ActionReceiver = actionReceiver;
            m_HeuristicProvider = heuristicProvider;
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
            Profiler.BeginSample("VectorActuator.OnActionReceived");
            m_ActionBuffers = actionBuffers;
            m_ActionReceiver.OnActionReceived(m_ActionBuffers);
            Profiler.EndSample();
        }

        public void Heuristic(in ActionBuffers actionBuffersOut)
        {
            Profiler.BeginSample("VectorActuator.Heuristic");
            m_HeuristicProvider?.Heuristic(actionBuffersOut);
            Profiler.EndSample();
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

        /// <inheritdoc />
        public virtual BuiltInActuatorType GetBuiltInActuatorType()
        {
            return BuiltInActuatorType.VectorActuator;
        }
    }
}
