using System.Collections.Generic;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents
{
    public interface ITeamManager
    {
        void RegisterAgent(Agent agent);
        // TODO not sure this is all the info we need, maybe pass a class/struct instead.
        void OnAgentDone(Agent agent, Agent.DoneReason doneReason, List<ISensor> sensors);
    }
}
