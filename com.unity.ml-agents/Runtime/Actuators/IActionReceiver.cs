using System;
using System.Linq;
using UnityEngine;

namespace TransformsAI.MicroMLAgents.Actuators
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

        /// <summary>
        /// Construct an <see cref="ActionBuffers"/> instance with the continuous and discrete actions that will
        /// be used.
        /// /// </summary>
        /// <param name="continuousActions">The continuous actions to send to an <see cref="IActionReceiver"/>.</param>
        /// <param name="discreteActions">The discrete actions to send to an <see cref="IActionReceiver"/>.</param>
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
        /// Construct an <see cref="ActionBuffers"/> instance with <see cref="ActionSpec"/>. All values are initialized to zeros.
        /// /// </summary>
        /// <param name="actionSpec">The <see cref="ActionSpec"/>  to send to an <see cref="IActionReceiver"/>.</param>
        public ActionBuffers(ActionSpec actionSpec)
            : this(new ActionSegment<float>(new float[actionSpec.NumContinuousActions]),
                new ActionSegment<int>(new int[actionSpec.NumDiscreteActions]))
        { }

        /// <summary>
        /// Create an <see cref="ActionBuffers"/> instance with ActionSpec and all actions stored as a float array.
        /// </summary>
        /// <param name="actionSpec"><see cref="ActionSpec"/> of the <see cref="ActionBuffers"/></param>
        /// <param name="actions">The float array of all actions, including discrete and continuous actions.</param>
        /// <returns>An <see cref="ActionBuffers"/> instance initialized with a <see cref="ActionSpec"/> and a float array.</returns>
        internal static ActionBuffers FromActionSpec(ActionSpec actionSpec, float[] actions)
        {
            if (actions == null)
            {
                return ActionBuffers.Empty;
            }

            Debug.Assert(actions.Length == actionSpec.NumContinuousActions + actionSpec.NumDiscreteActions,
                $"The length of '{nameof(actions)}' does not match the total size of ActionSpec.\n" +
                $"{nameof(actions)}.Length: {actions.Length}\n" +
                $"{nameof(actionSpec)}: {actionSpec.NumContinuousActions + actionSpec.NumDiscreteActions}");

            ActionSegment<float> continuousActionSegment = ActionSegment<float>.Empty;
            ActionSegment<int> discreteActionSegment = ActionSegment<int>.Empty;
            int offset = 0;
            if (actionSpec.NumContinuousActions > 0)
            {
                continuousActionSegment = new ActionSegment<float>(actions, 0, actionSpec.NumContinuousActions);
                offset += actionSpec.NumContinuousActions;
            }
            if (actionSpec.NumDiscreteActions > 0)
            {
                int[] discreteActions = new int[actionSpec.NumDiscreteActions];
                for (var i = 0; i < actionSpec.NumDiscreteActions; i++)
                {
                    discreteActions[i] = (int)actions[i + offset];
                }
                discreteActionSegment = new ActionSegment<int>(discreteActions);
            }

            return new ActionBuffers(continuousActionSegment, discreteActionSegment);
        }

        /// <summary>
        /// Clear the <see cref="ContinuousActions"/> and <see cref="DiscreteActions"/> segments to be all zeros.
        /// </summary>
        public void Clear()
        {
            ContinuousActions.Clear();
            DiscreteActions.Clear();
        }

        /// <summary>
        /// Check if the <see cref="ActionBuffers"/> is empty.
        /// </summary>
        /// <returns>Whether the buffers are empty.</returns>
        public bool IsEmpty()
        {
            return ContinuousActions.IsEmpty() && DiscreteActions.IsEmpty();
        }

        /// <summary>
        /// Indicates whether the current ActionBuffers is equal to another ActionBuffers.
        /// </summary>
        /// <param name="obj">An ActionBuffers to compare with this ActionBuffers.</param>
        /// <returns>true if the current ActionBuffers is equal to the other parameter; otherwise, false.</returns>
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

        /// <summary>
        /// Computes the hash code of the ActionBuffers.
        /// </summary>
        /// <returns>A hash code for the current ActionBuffers.</returns>
        public override int GetHashCode()
        {
            unchecked
            {
                return (ContinuousActions.GetHashCode() * 397) ^ DiscreteActions.GetHashCode();
            }
        }
    }
}
