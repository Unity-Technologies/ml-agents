namespace MLAgents.RewardProvider
{
    /// <summary>
    /// A typed reward provider that provides easy, typed access to RewardProvider implementations.
    /// Subclasses should
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class TypedRewardProviderComponent<T> : BaseRewardProviderComponent
    where T : IRewardProvider, new()
    {
        T m_TypedRewardProvider = new T();

        public T GetTypedRewardProvider()
        {
            return m_TypedRewardProvider;
        }

        public override IRewardProvider GetRewardProvider()
        {
            return m_TypedRewardProvider;
        }
    }
}
