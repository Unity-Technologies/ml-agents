using System.Collections.Generic;
using MLAgents.Sensor;

namespace MLAgents
{
    public struct ExperienceInfo
    {
        public AgentInfo agentInfo;
        public List<ISensor> sensors;
    }

    public interface IExperienceWriter
    {
        // TODO Start() and Stop()
        void Record(ExperienceInfo expInfo);
    }
}
