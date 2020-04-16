namespace Unity.MLAgents
{
    internal static class EpisodeIdCounter
    {
        static int s_Counter;
        public static int GetEpisodeId()
        {
            return s_Counter++;
        }
    }
}
