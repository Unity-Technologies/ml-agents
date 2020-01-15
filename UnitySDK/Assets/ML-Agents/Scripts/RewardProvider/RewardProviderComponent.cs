using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace MLAgents.RewardProvider
{
    /// <summary>
    /// The base class for all reward provider components.
    /// </summary>
    public class RewardProviderComponent : MonoBehaviour, IRewardProvider
    {

#if UNITY_EDITOR
        [Range(1, 100)]
        [Tooltip("The sample rate of the reward to display in the UI.  5 means it samples every 5 frames.")]
        public int RewardSampleRate = 20;
#endif
        /// <summary>
        /// The reward that is accumulated between Agent steps.
        /// </summary>
        float m_IncrementalReward;

        /// <summary>
        /// The Reward that is accumulated between Agent episodes.
        /// </summary>
        float m_CumulativeReward;
        
        /// <summary>
        /// Resets the step reward and possibly the episode reward for the agent.
        /// </summary>
        public void ResetReward(bool done = false)
        {
            m_IncrementalReward = 0f;
            if (done)
            {
                m_CumulativeReward = 0f;
            }
            
#if UNITY_EDITOR
            InternalResetReward();
#endif
        }

        /// <summary>
        /// Overrides the current step reward of the agent and updates the episode
        /// reward accordingly.
        /// </summary>
        /// <param name="reward">The new value of the reward.</param>
        public void SetReward(float reward)
        {
            m_CumulativeReward += reward - m_IncrementalReward;
            m_IncrementalReward = reward;
        }

        /// <summary>
        /// Increments the step and episode rewards by the provided value.
        /// </summary>
        /// <param name="increment">Incremental reward value.</param>
        public void AddReward(float increment)
        {
            m_IncrementalReward += increment;
            m_CumulativeReward += increment;
        }
        
        public float GetIncrementalReward()
        {
            return m_IncrementalReward;
        }

        public float GetCumulativeReward()
        {
            return m_CumulativeReward;
        }

        public virtual void RewardStep()
        {
            
        }
        

#if UNITY_EDITOR
        public AnimationCurve rewardCurve = new AnimationCurve();
#endif
        
#if UNITY_EDITOR
        void InternalResetReward()
        {
            if (Time.frameCount % RewardSampleRate != 0)
                return;
            var keyframe = new Keyframe
            {
                time = Time.realtimeSinceStartup,
                value = m_CumulativeReward,
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
