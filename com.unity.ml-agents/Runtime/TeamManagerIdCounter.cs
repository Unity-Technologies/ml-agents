using System.Threading;

namespace Unity.MLAgents
{
    internal static class TeamManagerIdCounter
    {
        static int s_Counter;
        public static int GetTeamManagerId()
        {
            return Interlocked.Increment(ref s_Counter); ;
        }
    }
}
