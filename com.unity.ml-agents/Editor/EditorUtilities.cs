using UnityEngine;

namespace MLAgents.Editor
{
    public static class EditorUtilities
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
