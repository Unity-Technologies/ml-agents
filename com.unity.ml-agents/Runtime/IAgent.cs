using System.Collections.Generic;
using TransformsAI.MicroMLAgents.Actuators;
using TransformsAI.MicroMLAgents.Sensors;

namespace TransformsAI.MicroMLAgents
{
    public interface IAgent
    {
        // Sensors
        public ObservationSpec[] SensorObservationSpecs { get; }
        int WriteObservation(ObservationWriter writer, int sensorIndex);

        // Actions
        // This can easily be initialized as `new ActionBuffers(this.ActionSpec)`
        public ActionBuffers ActionBuffer { get; set; }
        public ActionSpec ActionSpec { get; }

        // This holds the memories that the agent can use to store information when using recurrent networks.
        // This list will modified by the inference system.
        // Implementer is responsible for clearing the memories when resetting the agent.
        public List<float> Memory { get; }
        bool[] DiscreteActionMasks { get; }

    }
}

