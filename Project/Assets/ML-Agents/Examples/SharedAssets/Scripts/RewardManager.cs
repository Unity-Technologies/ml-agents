using System;
using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public class RewardManager : MonoBehaviour
{
    [Serializable]
    public class Reward
    {
        public string rewardKey;
//        [Range(.01f, .05f)]
        public float rewardScalar = .01f;
//        public float rewardScalar;
        public float rewardThisStep;
        public float cumulativeThisEpisode;
        public float cumulativeThisSession;

        public float maxRewardThisSession;
//        public Reward(string k)
//        public Reward()
//        {
////            rewardKey = k;
//            rewardScalar = .01f;
//        }
    }

    private Agent m_thisAgent;
    public List<Reward> rewardsList = new List<Reward>();
    public Dictionary<string, Reward> rewardsDict = new Dictionary<string, Reward>();
    public float maxSteps;
    private void OnEnable()
//    private void Awake()
    {
        m_thisAgent = GetComponent<Agent>();
        maxSteps = m_thisAgent.MaxStep;
        foreach (var item in rewardsList)
        {
            if (rewardsDict.ContainsKey(item.rewardKey)) return; //don't need to add
            rewardsDict.Add(item.rewardKey, item);
        }
    }

//    public void AddReward(Reward r)
//    {
//        if (rewardsDict.ContainsKey(r.rewardKey)) return; //don't need to add
//        rewardsDict.Add(r.rewardKey, r);
//    }
    
//    public void AddReward(string rewardKey)
//    {
//        if (rewardsDict.ContainsKey(rewardKey)) return; //don't need to add
//        Reward newReward = new Reward(rewardKey);
//        rewardsDict.Add(rewardKey, newReward);
//        rewardsList.Add(newReward);
//    }

    //Add new rewards
    public void UpdateReward(string key, float rawVal)
    {
        float val = rawVal * rewardsDict[key].rewardScalar;
        rewardsDict[key].maxRewardThisSession =1/maxSteps;
        rewardsDict[key].rewardThisStep = val;
        rewardsDict[key].cumulativeThisEpisode += val;
        rewardsDict[key].cumulativeThisSession += val;
        m_thisAgent.AddReward(val);
    }

//    //Add new rewards
//    public void UpdateReward(string key, float val)
//    {
//        rewardsDict[key].rewardThisStep = val;
//        rewardsDict[key].cumulativeThisEpisode += val;
//        rewardsDict[key].cumulativeThisSession += val;
//        m_thisAgent.AddReward(val);
//    }

    //Resets cumulative episode reward
    public void ResetEpisodeRewards()
    {
        foreach (var item in rewardsDict)
        {
            item.Value.rewardThisStep = 0;
            item.Value.cumulativeThisEpisode = 0;
        }
    }
    
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
