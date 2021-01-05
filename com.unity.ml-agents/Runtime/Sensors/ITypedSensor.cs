namespace Unity.MLAgents.Sensors
{

    /// <summary>
    /// The SensorType flag of the observation
    /// </summary>
    [System.Flags]
    public enum SensorType
    {
        Observation = 0,
        Goal = 1,
        Reward = 2,
        Message = 3,
    }


    /// <summary>
    /// Sensor interface for sensors with variable types.
    /// </summary>
    public interface ITypedSensor
    {
        /// <summary>
        /// Returns the SensorType enum corresponding to the type of the sensor.
        /// </summary>
        /// <returns>The SensorType enum</returns>
        SensorType GetSensorType();
    }
}
