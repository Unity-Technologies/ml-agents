using UnityEngine;

namespace MLAgents.RewardProvider
{
    /// <summary>
    /// A typed reward provider that provides easy, typed access to RewardProvider implementations.
    /// Subclasses should
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class BaseRewardProviderComponent<T> : MonoBehaviour
    where T : IRewardProvider, new()
    {
        T m_RewardProvider = new T();

        public T GetRewardProvider()
        {
            return m_RewardProvider;
        }
    }
}
