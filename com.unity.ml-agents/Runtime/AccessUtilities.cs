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

        internal static void LogUnableToUpdate()
        {
            Debug.Log("Unable to update property at this time.");
        }

        /// <summary>
        /// Update the target to the value if modifications are allowed. If not, a warning is logged.
        /// </summary>
        /// <param name="target"></param>
        /// <param name="value"></param>
        /// <typeparam name="T"></typeparam>
        public static void SetPropertyIfAllowed<T>(ref T target, T value)
        {
            if (CanUpdateModelProperties())
            {
                target = value;
            }
            else
            {
                LogUnableToUpdate();
            }
        }
    }
}
