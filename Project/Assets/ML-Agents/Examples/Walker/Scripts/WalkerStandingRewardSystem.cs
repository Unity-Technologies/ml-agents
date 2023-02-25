using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public class WalkerStandingRewardSystem : MonoBehaviour
{
    public WalkerASEAgent agent;
    public float RewardScale = 1.0f;

    void OnEnable()
    {
        Academy.Instance.AgentPreStep += AddStepReward;
    }

    void AddStepReward(int stepCount)
    {
        if (stepCount % agent.DecisionPeriod == 0)
        {
            float rewardScalingFactor = (float)agent.DecisionPeriod / agent.MaxStep;

            var balance = Mathf.Clamp(agent.GetChestBalance(), 0f, 1.0f);

            var rootBalance = Mathf.Clamp(agent.GetRootBalance(), 0f, 1.0f);

            var height = ZeroClamp(agent.GetRootHeightFromGround(), 0f, agent.StartHeight) / agent.StartHeight;

            agent.AddReward(balance * rootBalance * height * rewardScalingFactor * RewardScale);
        }
    }

    static float ZeroClamp(float value, float min, float max)
    {
        return value > max ? 0f : Mathf.Clamp(value, min, max);
    }
}
