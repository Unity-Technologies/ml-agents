using System;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;

namespace MLAgents.RewardProvider
{
    public class LowLevelRewardProviderComponent : MonoBehaviour
    {
        LowLevelRewardProvider m_RewardProvider = new LowLevelRewardProvider();
        public AnimationCurve rewardCurve = new AnimationCurve();

        public LowLevelRewardProvider GetRewardProvider()
        {
            return m_RewardProvider;
        }

        public virtual void Start()
        {
            GetRewardProvider().OnRewardProviderReset += RewardReset;
        }

        void RewardReset(float reward)
        {
#if UNITY_EDITOR
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
#endif
        }
    }
}
