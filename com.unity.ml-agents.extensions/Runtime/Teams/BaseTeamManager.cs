using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

namespace Teams
{
    // TODO abstract? inherit from MonoBehavior?
    public class BaseTeamManager : ITeamManager
    {
        public void RegisterAgent(Agent agent)
        {
            throw new System.NotImplementedException();
        }

        public void OnAgentDone(Agent agent, Agent.DoneReason doneReason, List<ISensor> sensors)
        {
            // Possible implementation - save reference to Agent's IPolicy so that we can repeatedly
            // call IPolicy.RequestDecision on behalf of the Agent after it's dead
            // If so, we'll need dummy sensor impls with the same shape as the originals.
            throw new System.NotImplementedException();
        }

        public void AddTeamReward(float reward)
        {

        }

    }
}
