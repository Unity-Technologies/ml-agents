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
        /// <summary>
        /// Default Sensor type if it cannot be determined.
        /// </summary>
        Unknown = 0,
        /// <summary>
        /// The Vector sensor used by the agent.
        /// </summary>
        VectorSensor = 1,
        /// <summary>
        /// The Stacking Sensor type. NOTE: StackingSensor actually returns the wrapped sensor's type.
        /// </summary>
        StackingSensor = 2,
        /// <summary>
        /// The RayPerception Sensor types, both 3D and 2D.
        /// </summary>
        RayPerceptionSensor = 3,
        /// <summary>
        /// The observable attribute sensor type.
        /// </summary>
        ReflectionSensor = 4,
        /// <summary>
        /// Sensors that use the Camera for observations.
        /// </summary>
        CameraSensor = 5,
        /// <summary>
        /// Sensors that use RenderTextures for observations.
        /// </summary>
        RenderTextureSensor = 6,
        /// <summary>
        /// Sensors that use buffers or tensors for observations.
        /// </summary>
        BufferSensor = 7,
        /// <summary>
        /// The sensors that observe properties of rigid bodies.
        /// </summary>
        PhysicsBodySensor = 8,
        /// <summary>
        /// The sensors that observe Match 3 boards.
        /// </summary>
        Match3Sensor = 9,
        /// <summary>
        /// Sensors that break down the world into a grid of colliders to observe an area at a pre-defined granularity.
        /// </summary>
        GridSensor = 10
    }

    /// <summary>
    /// Interface for sensors that are provided as part of ML-Agents.
    /// User-implemented sensors don't need to use this interface.
    /// </summary>
    internal interface IBuiltInSensor
    {
        /// <summary>
        /// Return the corresponding BuiltInSensorType for the sensor.
        /// </summary>
        /// <returns>A BuiltInSensorType corresponding to the sensor.</returns>
        BuiltInSensorType GetBuiltInSensorType();
    }
}
