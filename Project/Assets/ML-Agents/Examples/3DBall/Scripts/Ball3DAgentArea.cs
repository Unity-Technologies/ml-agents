using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using UnityEngine.Serialization;


public class Ball3DAgentArea : MonoBehaviour
{
    [HideInInspector]
    public List<GameObject> actorObjs;
    
    public GameObject prefab;

    [Tooltip("Number of Parallel Environments. ")]
    public int numberOfParallel = 1;
    [Tooltip("Number of maximum steps the agent can take in the environment. ")]
    public int maxStep = 100;
    [Tooltip("Specifies which reward function to use. For all environments")]
    public Ball3DRewardType rewardType;
    public float agentSpacing = 7.5f; 

    public int numPerRow = 6;
    
    EnvironmentParameters m_ResetParams;
    
    public void Start()
    {
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        actorObjs = new List<GameObject>();
        AreaReset();
    }


    public void MakeNewAgent(int idx)
    {
        float xpos = (idx % numPerRow) * agentSpacing;
        float ypos = (idx / numPerRow) * agentSpacing;
        var pos = new Vector3(xpos, ypos, 5f);
        actorObjs.Add(Instantiate(prefab, pos, Quaternion.identity));
    }

    public void update_agents()
    {
        foreach (var actor in actorObjs)
        {
            Ball3DMultiAgent agent = actor.GetComponent<Ball3DMultiAgent>();
            agent.m_RewardType = rewardType;
            agent.setMaxStep(maxStep);
        }
    }
    public void AreaReset()
    {
        int currentN = actorObjs.Count;
        if (numberOfParallel < currentN){
            for (int  i=currentN-1; i >= numberOfParallel; i--)
            {
                DestroyImmediate(actorObjs[i]);
            }
            actorObjs.RemoveRange(numberOfParallel, currentN-numberOfParallel);
        }
        else if (numberOfParallel > currentN)
        {
            for (int i=currentN; i < numberOfParallel; i++)
            {
                MakeNewAgent(i);
            }
        }
    }
    public void FixedUpdate()
    {
        int N = (int)m_ResetParams.GetWithDefault("numParallel", numberOfParallel);
        int newStep = (int)m_ResetParams.GetWithDefault("maxStep", maxStep);
        int rtype = (int)m_ResetParams.GetWithDefault("rewardType", -1);
        Ball3DRewardType rt = rewardType;
        bool changed = false;
        if (N != numberOfParallel)
        {
            changed = true;
            numberOfParallel = N;
        }
        if (newStep != maxStep)
        {
            changed = true;
            maxStep = newStep;
        }
        if (rtype == 0)
        {
            rt = Ball3DRewardType.Time;
        } 
        else if (rtype == 1)
        {
            rt = Ball3DRewardType.Distance;
        }
        else if (rtype == 2)
        {
            rt = Ball3DRewardType.Power;
        }

        if (rt != rewardType)
        {
            changed = true;
            rewardType = rt;
        }
        if (changed)
        {
            AreaReset();
            update_agents();
        }
    }
}
