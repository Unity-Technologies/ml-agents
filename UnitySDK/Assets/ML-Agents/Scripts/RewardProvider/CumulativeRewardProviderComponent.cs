using System;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;

namespace MLAgents.RewardProvider
{
    public class CumulativeRewardProviderComponent : RewardProviderComponent
    {
        CumulativeRewardProvider m_RewardProvider = new CumulativeRewardProvider();
        
#if UNITY_EDITOR
        public AnimationCurve rewardCurve = new AnimationCurve();
#endif

        public override IRewardProvider GetRewardProvider()
        {
            return m_RewardProvider;
        }

#if UNITY_EDITOR
        public void Start()
        {
            m_RewardProvider.OnRewardProviderReset += RewardReset;
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
#endif
    }
}
