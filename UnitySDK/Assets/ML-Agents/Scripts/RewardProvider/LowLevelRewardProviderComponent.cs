using System;
using UnityEditor;
using UnityEngine;

namespace MLAgents.RewardProvider
{
    public class LowLevelRewardProviderComponent : MonoBehaviour
    {
        LowLevelRewardProvider m_RewardProvider;
        public AnimationCurve rewardCurve = new AnimationCurve();

        public LowLevelRewardProvider GetRewardProvider()
        {
            return m_RewardProvider;
        }
        public virtual void Start()
        {
            m_RewardProvider = new LowLevelRewardProvider();
            GetRewardProvider().OnRewardProviderReset += RewardReset;
        }

        void RewardReset(float reward)
        {
            var keyframe = new Keyframe
            {
                time = Time.realtimeSinceStartup,
                value = reward,
                inTangent = 0.0f,
                outTangent = 0.0f
            };
            var index = rewardCurve.AddKey(keyframe);
            AnimationUtility.SetKeyLeftTangentMode(rewardCurve, index, AnimationUtility.TangentMode.Linear);
            AnimationUtility.SetKeyRightTangentMode(rewardCurve, index, AnimationUtility.TangentMode.Linear);
        }
    }
}
