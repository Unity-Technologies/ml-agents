using System.Collections.Generic;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents
{
    public interface ITeamManager
    {
        int GetId();

        void RegisterAgent(Agent agent);
    }
}
