namespace Unity.MLAgents.Sensors
{

    /// <summary>
    /// The SensorType enum of the observation
    /// </summary>
    internal enum SensorType
    {
        Observation = 0,
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
        /// Returns the SensorType enum corresponding to the type of the sensor.
        /// </summary>
        /// <returns>The SensorType enum</returns>
        SensorType GetSensorType();
    }
}
