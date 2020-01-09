using UnityEngine;

namespace MLAgents.RewardProvider
{
    /// <summary>
    /// The abstract base class for all reward provider components.
    /// </summary>
    public abstract class RewardProviderComponent : MonoBehaviour
    {
        /// <summary>
        /// Returns the IRewardProvider held by this component.
        /// </summary>
        /// <returns>An instance of IRewardProvider</returns>
        public abstract IRewardProvider GetRewardProvider();
    }
}
