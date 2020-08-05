namespace Unity.MLAgents.Actuators
{
    public interface IActionSpec
    {

        /// <summary>
        /// The number of actions for a Continuous action space.
        /// </summary>
        int NumContinuousActions { get; }

        /// <summary>
        /// The number of branches for a Discrete action space.
        /// </summary>
        int NumDiscreteActions { get; }

        /// <summary>
        /// Get the total number of Discrete Actions that can be taken by calculating the Sum
        /// of all of the Discrete Action branch sizes.
        /// </summary>
        int SumOfDiscreteBranchSizes { get; }
    }
}
