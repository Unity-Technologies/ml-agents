using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MujocoUnity;
using UnityEngine;

public class OpenAIAntAgent : MujocoAgent {

    public override void AgentReset()
    {
        base.AgentReset();

        // set to true this to show monitor while training
        Monitor.SetActive(true);

        StepRewardFunction = StepReward_Ant101;
        TerminateFunction = Terminate_Never;
        ObservationsFunction = Observations_Default;

        BodyParts["pelvis"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="torso_geom");
        base.SetupBodyParts();
    }


    public override void AgentOnDone()
    {

    }
    void Observations_Default()
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


    float StepReward_Ant101()
    {
        float velocity = GetVelocity();
        float effort = GetEffort();
        var effortPenality = 1e-3f * (float)effort;
        var jointsAtLimitPenality = GetJointsAtLimitPenality() * 4;

        var reward = velocity
            - jointsAtLimitPenality
            -effortPenality;
        if (ShowMonitor) {
            var hist = new []{reward,velocity, -jointsAtLimitPenality, -effortPenality}.ToList();
            Monitor.Log("rewardHist", hist, MonitorType.hist);
        }

        return reward;
    }
}
