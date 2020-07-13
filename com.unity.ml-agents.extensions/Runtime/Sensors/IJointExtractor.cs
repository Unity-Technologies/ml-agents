using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    public interface IJointExtractor
    {
        int NumObservations(PhysicsSensorSettings settings);
        int Write(PhysicsSensorSettings settings, ObservationWriter writer, int offset);
    }
}
