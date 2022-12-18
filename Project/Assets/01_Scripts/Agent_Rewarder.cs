using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Agent_Rewarder : MonoBehaviour
{
    //Controls total reward sent to agent
    private int rewardTotal;

    //Speed Optimization
    [SerializeField] private bool EncourageSpeed;
    [SerializeField] private int MaxSteps;
    [SerializeField] private int CurrentSteps;


    //Collission Target
    [SerializeField] private bool GoToTarget;
    [SerializeField] private Transform Target;          //Potentially change this to a collider?

    //List of things to Touch
    //List of things to Avoid
    //List of Allies?


    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void FixedUpdate()
    {
        //Increment Step Counter
        CurrentSteps++;
    }

    public float GetAgentReward()
    {
        float _reward = 0.0f;

        //This is the part where we calculate all the reward vectores UwU....
        if(EncourageSpeed)
        {
            _reward += SpeedReward(1.0f);
        }

        return _reward;
    }



    private float SpeedReward(float _weight)
    {
        //Add Weighting?
        float _reward = (CurrentSteps / MaxSteps) * _weight;
        return _reward;
    }

    private float TargetReward(float _weight)
    {
        float _reward = (_weight * 1.0f);
        return _reward;
    }



    void OnCollisionEnter(Collision other)
    {
        if(GoToTarget)
        {
            //Hit the thing I waant to hit
            if(other.gameObject == Target.gameObject)
            {
                //Reward on hit!

            }
        }
    }

}
