namespace Unity.MLAgents.Sensors
{

    /// <summary>
    /// The ObservationType enum of the Sensor.
    /// </summary>
    internal enum ObservationType
    {
        Default = 0,
        Goal = 1,
        Reward = 2,
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
