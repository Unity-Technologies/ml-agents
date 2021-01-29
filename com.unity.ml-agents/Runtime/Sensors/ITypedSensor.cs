namespace Unity.MLAgents.Sensors
{

    /// <summary>
    /// The ObservationType enum of the Sensor.
    /// </summary>
    internal enum ObservationType
    {
        // Collected observations are generic.
        Default = 0,
        // Collected observations contain goal information.
        Goal = 1,
        // Collected observations contain reward information.
        Reward = 2,
        // Collected observations are messages from other agents.
        Message = 3,
    }


    /// <summary>
    /// Sensor interface for sensors with variable types.
    /// </summary>
    internal interface ITypedSensor
    {
        /// <summary>
        /// Returns the ObservationType enum corresponding to the type of the sensor.
        /// </summary>
        /// <returns>The ObservationType enum</returns>
        ObservationType GetObservationType();
    }
}
