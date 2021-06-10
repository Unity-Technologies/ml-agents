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
    public Ball3DRewardType rewardType = Ball3DRewardType.Time;
    public float agentSpacing = 7.5f; 

    public int numPerRow = 6;

    public int decisionFrequency = 5;
    
    EnvironmentParameters m_ResetParams;
    
    public void Awake()
    {
        Academy.Instance.OnEnvironmentReset += UpdateEnvs;
    }

    public void Start()
    {
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        actorObjs = new List<GameObject>();
        AreaReset();
        update_agents();
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
            Ball3DMultiAgent agent = actor.GetComponentInChildren<Ball3DMultiAgent>();    
            agent.m_RewardType = rewardType;
            agent.setMaxStep(maxStep * decisionFrequency);
            DecisionRequester dr = agent.GetComponent<DecisionRequester>();
            dr.DecisionPeriod = decisionFrequency;
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
    public void UpdateEnvs()
    {
        int N = (int)m_ResetParams.GetWithDefault("numParallel", numberOfParallel);
        int newStep = (int)m_ResetParams.GetWithDefault("maxStep", maxStep);
        int rtype = (int)m_ResetParams.GetWithDefault("rewardType", -1);
        int df = (int)m_ResetParams.GetWithDefault("decisionFreq", decisionFrequency);
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
        if (df != decisionFrequency)
        {
            changed = true;
            decisionFrequency = df;
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
        }
        update_agents();
    }
}
