namespace Unity.MLAgents.Actuators
{
    /// <summary>
    /// Abstraction that facilitates the execution of actions.
    /// </summary>
    public interface IActuator : IActionReceiver
    {
        /// <summary>
        /// The specification of the actions for this IActuator.
        /// </summary>
        /// <seealso cref="ActionSpec"/>
        ActionSpec ActionSpec { get; }

        /// <summary>
        /// Gets the name of this IActuator which will be used to sort it.
        /// </summary>
        /// <returns></returns>
        string Name { get; }

        /// <summary>
        /// Resets the internal state of the actuator. This is called at the end of an Agent's episode.
        /// Most implementations can leave this empty.
        /// </summary>
        void ResetData();
    }

    /// <summary>
    /// Helper methods to be shared by all classes that implement <see cref="IActuator"/>.
    /// </summary>
    public static class IActuatorExtensions
    {
        /// <summary>
        /// Returns the number of discrete branches + the number of continuous actions.
        /// </summary>
        /// <param name="actuator"></param>
        /// <returns></returns>
        public static int TotalNumberOfActions(this IActuator actuator)
        {
            return actuator.ActionSpec.NumContinuousActions + actuator.ActionSpec.NumDiscreteActions;
        }
    }
}
