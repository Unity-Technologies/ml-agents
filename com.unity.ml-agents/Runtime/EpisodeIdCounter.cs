using UnityEngine;

namespace Unity.MLAgents
{
    internal static class EpisodeIdCounter
    {
        static int s_Counter;
        public static int GetEpisodeId()
        {
            return s_Counter++;
        }
#if UNITY_EDITOR
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStaticsOnLoad()
        {
            s_Counter = 0;
        }
#endif
    }
}
