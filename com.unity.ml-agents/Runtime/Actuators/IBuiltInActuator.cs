namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// Identifiers for "built in" actuator types.
    /// These are only used for analytics, and should not be used for any runtime decisions.
    ///
    /// NOTE: Do not renumber these, since the values are used for analytics. Renaming is allowed though.
    /// </summary>
    public enum BuiltInActuatorType
    {
        /// <summary>
        /// Default Sensor type if it cannot be determined.
        /// </summary>
        Unknown = 0,

        /// <summary>
        /// VectorActuator used by the Agent
        /// </summary>
        AgentVectorActuator = 1,

        /// <summary>
        /// Corresponds to <see cref="VectorActuator"/>
        /// </summary>
        VectorActuator = 2,

        /// <summary>
        /// Corresponds to the Match3Actuator in com.unity.ml-agents.extensions.
        /// </summary>
        Match3Actuator = 3,

        /// <summary>
        /// Corresponds to the InputActionActuator in com.unity.ml-agents.extensions.
        /// </summary>
        InputActionActuator = 4,
    }

    /// <summary>
    /// Interface for actuators that are provided as part of ML-Agents.
    /// User-implemented actuators don't need to use this interface.
    /// </summary>
    internal interface IBuiltInActuator
    {
        /// <summary>
        /// Return the corresponding BuiltInActuatorType for the actuator.
        /// </summary>
        /// <returns>A BuiltInActuatorType corresponding to the actuator.</returns>
        BuiltInActuatorType GetBuiltInActuatorType();
    }
}
