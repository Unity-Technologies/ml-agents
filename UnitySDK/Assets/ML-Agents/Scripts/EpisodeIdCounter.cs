namespace MLAgents
{
    public static class EpisodeIdCounter
    {
        private static int Counter;
        public static int GetEpisodeId()
        {
            return Counter++;
        }
    }
}