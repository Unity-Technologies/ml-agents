using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Teams
{
    public class BaseTeamManager : ITeamManager
    {
        readonly string m_Id = System.Guid.NewGuid().ToString();

        public virtual void RegisterAgent(Agent agent)
        {
            throw new System.NotImplementedException();
        }

        public virtual void OnAgentDone(Agent agent, Agent.DoneReason doneReason, List<ISensor> sensors)
        {
            // Possible implementation - save reference to Agent's IPolicy so that we can repeatedly
            // call IPolicy.RequestDecision on behalf of the Agent after it's dead
            // If so, we'll need dummy sensor impls with the same shape as the originals.
            throw new System.NotImplementedException();
        }

        public virtual void AddTeamReward(float reward)
        {

        }

        public string GetId()
        {
            return m_Id;
        }

    }
}
