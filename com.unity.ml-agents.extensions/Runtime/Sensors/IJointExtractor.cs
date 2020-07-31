using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Interface for generating observations from a physical joint or constraint.
    /// </summary>
    public interface IJointExtractor
    {
        /// <summary>
        /// Determine the number of observations that would be generated for the particular joint
        /// using the provided PhysicsSensorSettings.
        /// </summary>
        /// <param name="settings"></param>
        /// <returns>Number of floats that will be written.</returns>
        int NumObservations(PhysicsSensorSettings settings);

        /// <summary>
        /// Write the observations to the ObservationWriter, starting at the specified offset.
        /// </summary>
        /// <param name="settings"></param>
        /// <param name="writer"></param>
        /// <param name="offset"></param>
        /// <returns>Number of floats that were written.</returns>
        int Write(PhysicsSensorSettings settings, ObservationWriter writer, int offset);
    }
}
