using System;
using UnityEngine;

namespace Unity.MLAgents
{
    public partial class Agent
    {
        /// <summary>
        /// Deprecated, use <see cref="WriteDiscreteActionMask"/> instead.
        /// </summary>
        /// <param name="actionMasker"></param>
        [Obsolete("CollectDiscreteActionMasks has been deprecated, please use WriteDiscreteActionMask.")]
        public virtual void CollectDiscreteActionMasks(DiscreteActionMasker actionMasker)
        {
        }

        /// <summary>
        /// Deprecated, use <see cref="Heuristic(in ActionBuffers)"/> instead.
        /// </summary>
        /// <param name="actionsOut"></param>
        [Obsolete("The float[] version of Heuristic has been deprecated, please use the ActionBuffers version instead.")]
        public virtual void Heuristic(float[] actionsOut)
        {
            Debug.LogWarning("Heuristic method called but not implemented. Returning placeholder actions.");
            Array.Clear(actionsOut, 0, actionsOut.Length);
        }

        /// <summary>
        /// Deprecated, use <see cref="OnActionReceived(ActionBuffers)"/> instead.
        /// </summary>
        /// <param name="vectorAction"></param>
        [Obsolete("The float[] version of OnActionReceived has been deprecated, please use the ActionBuffers version instead.")]
        public virtual void OnActionReceived(float[] vectorAction) { }

        /// <summary>
        /// Returns the last action that was decided on by the Agent.
        /// </summary>
        /// <returns>
        /// The last action that was decided by the Agent (or null if no decision has been made).
        /// </returns>
        /// <seealso cref="OnActionReceived(ActionBuffers)"/>
        [Obsolete("GetAction has been deprecated, please use GetStoredActionBuffers instead.")]
        public float[] GetAction()
        {
            var storedAction = m_Info.storedVectorActions;
            if (!storedAction.ContinuousActions.IsEmpty())
            {
                return storedAction.ContinuousActions.Array;
            }
            else
            {
                return Array.ConvertAll(storedAction.DiscreteActions.Array, x => (float)x);
            }
        }
    }
}
