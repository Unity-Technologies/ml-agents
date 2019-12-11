using System;
using UnityEngine;

namespace MLAgents.RewardProvider
{
    public class BaseRewardProviderComponent: MonoBehaviour
    {
        IRewardProvider m_RewardProvider;

        public virtual IRewardProvider GetRewardProvider()
        {
            return m_RewardProvider;
        }
    }
}
