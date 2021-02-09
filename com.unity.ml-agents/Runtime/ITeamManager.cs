namespace Unity.MLAgents
{
    public interface ITeamManager
    {
        int GetId();

        void RegisterAgent(Agent agent);

        void UnregisterAgent(Agent agent);
    }
}
