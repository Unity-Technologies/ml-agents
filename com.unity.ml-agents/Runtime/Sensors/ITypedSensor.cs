namespace Unity.MLAgents.Sensors
{

    /// <summary>
    /// The ObservationType enum of the Sensor.
    /// </summary>
    public enum ObservationType
    {
        // Collected observations are generic.
        Default,
        // Collected observations contain goal information.
        Goal,
        // Collected observations contain reward information.
        Reward,
        // Collected observations are messages from other agents.
        Message,
    }


    /// <summary>
    /// Sensor interface for sensors with variable types.
    /// </summary>
    public interface ITypedSensor
    {
        /// <summary>
        /// Returns the ObservationType enum corresponding to the type of the sensor.
        /// </summary>
        /// <returns>The ObservationType enum</returns>
        ObservationType GetObservationType();
    }
}
