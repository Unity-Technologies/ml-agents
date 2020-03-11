using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Methods for determining whether a property is modifiable at the current time.
    /// This is because some properties, namely ones that affect the inputs and outputs
    /// of the reinforcement learning model, cannot be updated once simulation has started.
    /// </summary>
    public static class AccessUtilities
    {
        /// <summary>
        /// Whether or not properties that affect the model can be updated at the current time.
        /// </summary>
        /// <returns></returns>
        public static bool CanUpdateModelProperties()
        {
            return !Application.isPlaying;
        }
    }
}
