namespace Unity.MLAgents
{
    public interface IMultiAgentGroup
    {
        int GetId();

        void RegisterAgent(Agent agent);

        void UnregisterAgent(Agent agent);
    }
}
