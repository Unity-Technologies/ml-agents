using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Teams
{
    public class BaseTeamManager : ITeamManager
    {
        readonly int m_Id = TeamManagerIdCounter.GetTeamManagerId();

        public virtual void OnAgentDone(Agent agent, Agent.DoneReason doneReason, List<ISensor> sensors)
        {
            // Possible implementation - save reference to Agent's IPolicy so that we can repeatedly
            // call IPolicy.RequestDecision on behalf of the Agent after it's dead
            // If so, we'll need dummy sensor impls with the same shape as the originals.
            agent.SendDoneToTrainer();
        }

        public virtual void RegisterAgent(Agent agent) { }

        public int GetId()
        {
            return m_Id;
        }
    }
}
