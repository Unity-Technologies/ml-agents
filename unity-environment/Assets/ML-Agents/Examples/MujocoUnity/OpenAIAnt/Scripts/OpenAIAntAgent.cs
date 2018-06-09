using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MujocoUnity;
using UnityEngine;
using MLAgents;

public class OpenAIAntAgent : MujocoAgent {

    public override void AgentReset()
    {
        base.AgentReset();

        // set to true this to show monitor while training
        Monitor.SetActive(true);

        StepRewardFunction = StepRewardAnt101;
        TerminateFunction = TerminateNever;
        ObservationsFunction = ObservationsDefault;

        BodyParts["pelvis"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="torso_geom");
        base.SetupBodyParts();
    }


    public override void AgentOnDone()
    {

    }
    void ObservationsDefault()
    {
        if (ShowMonitor) {
        }
        var pelvis = BodyParts["pelvis"];
        AddVectorObs(pelvis.velocity);
        AddVectorObs(pelvis.transform.forward); // gyroscope 
        AddVectorObs(pelvis.transform.up);
        
        AddVectorObs(MujocoController.SensorIsInTouch);
        MujocoController.JointRotations.ForEach(x=>AddVectorObs(x));
        AddVectorObs(MujocoController.JointVelocity);
    }


    float StepRewardAnt101()
    {
        float velocity = GetVelocity();
        float effort = GetEffort();
        var effortPenality = 1e-1f * (float)effort;
        var jointsAtLimitPenality = GetJointsAtLimitPenality() * 4;

        var reward = velocity
            - jointsAtLimitPenality
            -effortPenality;
        if (ShowMonitor) {
            var hist = new []{reward,velocity, -jointsAtLimitPenality, -effortPenality}.ToList();
            Monitor.Log("rewardHist", hist.ToArray());
        }

        return reward;
    }
}
