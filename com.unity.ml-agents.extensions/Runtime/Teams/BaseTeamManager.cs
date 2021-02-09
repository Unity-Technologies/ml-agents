namespace Unity.MLAgents.Extensions.Teams
{
    public class BaseTeamManager : ITeamManager
    {
        readonly int m_Id = TeamManagerIdCounter.GetTeamManagerId();


        public virtual void RegisterAgent(Agent agent) { }

        public int GetId()
        {
            return m_Id;
        }
    }
}
