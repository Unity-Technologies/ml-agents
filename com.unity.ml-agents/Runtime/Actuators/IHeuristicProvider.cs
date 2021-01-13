namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// Interface that allows objects to fill out an <see cref="ActionBuffers"/> data structure for controlling
    /// behavior of Agents or Actuators.
    /// </summary>
    public interface IHeuristicProvider
    {
        /// <summary>
        /// Method called on objects which are expected to fill out the <see cref="ActionBuffers"/> data structure.
        /// Object that implement this interface should be careful to be consistent in the placement of their actions
        /// in the <see cref="ActionBuffers"/> data structure.
        /// </summary>
        /// <param name="actionBuffersOut">The <see cref="ActionBuffers"/> data structure to be filled by the
        /// object implementing this interface.</param>
        void Heuristic(in ActionBuffers actionBuffersOut);
    }
}
