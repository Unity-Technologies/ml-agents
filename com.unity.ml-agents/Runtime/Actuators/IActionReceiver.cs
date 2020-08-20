using System;
using System.Linq;
using UnityEngine;

namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// A structure that wraps the <see cref="ActionSegment{T}"/>s for a particular <see cref="IActionReceiver"/> and is
    /// used when <see cref="IActionReceiver.OnActionReceived"/> is called.
    /// </summary>
    public readonly struct ActionBuffers
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
        /// Create an <see cref="ActionBuffers"/> instance with discrete actions stored as a float array.  This exists
        /// to achieve backward compatibility with the former Agent methods which used a float array for both continuous
        /// and discrete actions.
        /// </summary>
        /// <param name="discreteActions">The float array of discrete actions.</param>
        /// <returns>An <see cref="ActionBuffers"/> instance initialized with a <see cref="DiscreteActions"/>
        /// <see cref="ActionSegment{T}"/> initialized from a float array.</returns>
        public static ActionBuffers FromDiscreteActions(float[] discreteActions)
        {
            return new ActionBuffers(ActionSegment<float>.Empty, discreteActions == null ? ActionSegment<int>.Empty
                                : new ActionSegment<int>(Array.ConvertAll(discreteActions,
                                    x => (int)x)));
        }

        public ActionBuffers(float[] continuousActions, int[] discreteActions)
            : this(new ActionSegment<float>(continuousActions), new ActionSegment<int>(discreteActions)) { }

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

        /// <summary>
        /// Clear the <see cref="ContinuousActions"/> and <see cref="DiscreteActions"/> segments to be all zeros.
        /// </summary>
        public void Clear()
        {
            ContinuousActions.Clear();
            DiscreteActions.Clear();
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

        /// <summary>
        /// Packs the continuous and discrete actions into one float array.  The array passed into this method
        /// must have a Length that is greater than or equal to the sum of the Lengths of
        /// <see cref="ContinuousActions"/> and <see cref="DiscreteActions"/>.
        /// </summary>
        /// <param name="destination">A float array to pack actions into whose length is greater than or
        /// equal to the addition of the Lengths of this objects <see cref="ContinuousActions"/> and
        /// <see cref="DiscreteActions"/> segments.</param>
        public void PackActions(in float[] destination)
        {
            Debug.Assert(destination.Length >= ContinuousActions.Length + DiscreteActions.Length,
                $"argument '{nameof(destination)}' is not large enough to pack the actions into.\n" +
                $"{nameof(destination)}.Length: {destination.Length}\n" +
                $"{nameof(ContinuousActions)}.Length + {nameof(DiscreteActions)}.Length: {ContinuousActions.Length + DiscreteActions.Length}");

            var start = 0;
            if (ContinuousActions.Length > 0)
            {
                Array.Copy(ContinuousActions.Array,
                    ContinuousActions.Offset,
                    destination,
                    start,
                    ContinuousActions.Length);
                start = ContinuousActions.Length;
            }
            if (start >= destination.Length)
            {
                return;
            }

            if (DiscreteActions.Length > 0)
            {
                Array.Copy(DiscreteActions.Array,
                    DiscreteActions.Offset,
                    destination,
                    start,
                    DiscreteActions.Length);
            }
        }
    }

    /// <summary>
    /// An interface that describes an object that can receive actions from a Reinforcement Learning network.
    /// </summary>
    public interface IActionReceiver
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

    /// <summary>
    /// Helper methods to be shared by all classes that implement <see cref="IActionReceiver"/>.
    /// </summary>
    public static class ActionReceiverExtensions
    {
        /// <summary>
        /// Returns the number of discrete branches + the number of continuous actions.
        /// </summary>
        /// <param name="actionReceiver"></param>
        /// <returns></returns>
        public static int TotalNumberOfActions(this IActionReceiver actionReceiver)
        {
            return actionReceiver.ActionSpec.NumContinuousActions + actionReceiver.ActionSpec.NumDiscreteActions;
        }
    }
}
