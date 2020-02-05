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
        void Initialize(string demonstrationName, BrainParameters brainParameters, string brainName);
        void Record(ExperienceInfo expInfo);
        // TODO implement IDisposable and rename to Dispose?
        void Close();
    }
}
