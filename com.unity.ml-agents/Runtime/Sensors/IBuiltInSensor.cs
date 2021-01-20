namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// Identifiers for "built in" sensor types.
    /// These are only used for analytics, and should not be used for any runtime decisions.
    ///
    /// NOTE: Do not renumber these, since the values are used for analytics. Renaming is allowed though.
    /// </summary>
    public enum BuiltInSensorType
    {
        Unknown = 0,
        VectorSensor = 1,
        // Note that StackingSensor actually returns the wrapped sensor's type
        StackingSensor = 2,
        RayPerceptionSensor = 3,
        ReflectionSensor = 4,
        CameraSensor = 5,
        RenderTextureSensor = 6,
        BufferSensor = 7,
        PhysicsBodySensor = 8,
        Match3Sensor = 9,
        GridSensor = 10
    }

    /// <summary>
    /// Interface for sensors that are provided as part of ML-Agents.
    /// User-implemented sensors don't need to use this interface.
    /// </summary>
    public interface IBuiltInSensor
    {
        /// <summary>
        /// Return the corresponding BuiltInSensorType for the sensor.
        /// </summary>
        /// <returns>A BuiltInSensorType corresponding to the sensor.</returns>
        BuiltInSensorType GetBuiltInSensorType();
    }


}
