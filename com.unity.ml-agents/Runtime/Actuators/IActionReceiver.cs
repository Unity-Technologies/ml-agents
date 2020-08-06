using System;
using System.Linq;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// A structure that wraps the <see cref="ActionSegment{T}"/>s for a particular <see cref="IActionReceiver"/> and is
    /// used when <see cref="IActionReceiver.OnActionReceived"/> is called.
    /// </summary>
    internal readonly struct ActionBuffers
    {
        /// <summary>
        /// An empty action buffer.
        /// </summary>
        public static ActionBuffers Empty = new ActionBuffers(ActionSegment<float>.Empty, ActionSegment<int>.Empty);

        /// <summary>
        /// Holds the Continuous <see cref="ActionSegment{T}"/> to be used by an <see cref="IActionReceiver"/>.
        /// </summary>
        public ActionSegment<float> ContinuousActions { get; }

        /// <summary>
        /// Holds the Discrete <see cref="ActionSegment{T}"/> to be used by an <see cref="IActionReceiver"/>.
        /// </summary>
        public ActionSegment<int> DiscreteActions { get; }

        /// <summary>
        /// Construct an <see cref="ActionBuffers"/> instance with the continuous and discrete actions that will
        /// be used.
        /// </summary>
        /// <param name="continuousActions">The continuous actions to send to an <see cref="IActionReceiver"/>.</param>
        /// <param name="discreteActions">The discrete actions to send to an <see cref="IActionReceiver"/>.</param>
        public ActionBuffers(ActionSegment<float> continuousActions, ActionSegment<int> discreteActions)
        {
            ContinuousActions = continuousActions;
            DiscreteActions = discreteActions;
        }

        /// <inheritdoc cref="ValueType.Equals(object)"/>
        public override bool Equals(object obj)
        {
            if (!(obj is ActionBuffers))
            {
                return false;
            }

            var ab = (ActionBuffers)obj;
            return ab.ContinuousActions.SequenceEqual(ContinuousActions) &&
                ab.DiscreteActions.SequenceEqual(DiscreteActions);
        }

        /// <inheritdoc cref="ValueType.GetHashCode"/>
        public override int GetHashCode()
        {
            unchecked
            {
                return (ContinuousActions.GetHashCode() * 397) ^ DiscreteActions.GetHashCode();
            }
        }
    }

    /// <summary>
    /// An interface that describes an object that can receive actions from a Reinforcement Learning network.
    /// </summary>
    internal interface IActionReceiver
    {

        /// <summary>
        /// The specification of the Action space for this IActionReceiver.
        /// </summary>
        /// <seealso cref="ActionSpec"/>
        ActionSpec ActionSpec { get; }

        /// <summary>
        /// Method called in order too allow object to execute actions based on the
        /// <see cref="ActionBuffers"/> contents.  The structure of the contents in the <see cref="ActionBuffers"/>
        /// are defined by the <see cref="ActionSpec"/>.
        /// </summary>
        /// <param name="actionBuffers">The data structure containing the action buffers for this object.</param>
        void OnActionReceived(ActionBuffers actionBuffers);

        /// <summary>
        /// Implement `WriteDiscreteActionMask()` to modify the masks for discrete
        /// actions. When using discrete actions, the agent will not perform the masked
        /// action.
        /// </summary>
        /// <param name="actionMask">
        /// The action mask for the agent.
        /// </param>
        /// <remarks>
        /// When using Discrete Control, you can prevent the Agent from using a certain
        /// action by masking it with <see cref="IDiscreteActionMask.WriteMask"/>.
        ///
        /// See [Agents - Actions] for more information on masking actions.
        ///
        /// [Agents - Actions]: https://github.com/Unity-Technologies/ml-agents/blob/release_4_docs/docs/Learning-Environment-Design-Agents.md#actions
        /// </remarks>
        /// <seealso cref="IActionReceiver.OnActionReceived"/>
        void WriteDiscreteActionMask(IDiscreteActionMask actionMask);
    }
}
