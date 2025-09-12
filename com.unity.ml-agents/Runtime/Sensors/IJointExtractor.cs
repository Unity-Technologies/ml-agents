namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// Interface for generating observations from a physical joint or constraint.
    /// </summary>
    [UnityEngine.Scripting.APIUpdating.MovedFrom("Unity.MLAgents.Extensions.Sensors")]
    public interface IJointExtractor
    {
        /// <summary>
        /// Determine the number of observations that would be generated for the particular joint
        /// using the provided PhysicsSensorSettings.
        /// </summary>
        /// <param name="settings">The settings used to configure the physics sensor.</param>
        /// <returns>Number of floats that will be written.</returns>
        int NumObservations(PhysicsSensorSettings settings);

        /// <summary>
        /// Write the observations to the ObservationWriter, starting at the specified offset.
        /// </summary>
        /// <param name="settings">The settings used to configure the physics sensor.</param>
        /// <param name="writer">The writer to which observations are written.</param>
        /// <param name="offset">The starting index in the writer to begin writing observations.</param>
        /// <returns>Number of floats that were written.</returns>
        int Write(PhysicsSensorSettings settings, ObservationWriter writer, int offset);
    }
}
