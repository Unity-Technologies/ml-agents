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
        Unknown = 0,
        // VectorActuator used by the Agent
        AgentVectorActuator = 1,
        VectorActuator = 2,
        Match3Actuator = 3
    }

    /// <summary>
    /// Interface for actuators that are provided as part of ML-Agents.
    /// User-implemented actuators don't need to use this interface.
    /// </summary>
    public interface IBuiltInActuator
    {
        /// <summary>
        /// Return the corresponding BuiltInActuatorType for the actuator.
        /// </summary>
        /// <returns>A BuiltInActuatorType corresponding to the actuator.</returns>
        BuiltInActuatorType GetBuiltInActuatorType();
    }
}
