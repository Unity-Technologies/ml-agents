using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MujocoUnity;
using UnityEngine;
using MLAgents;

public class DeepMindWalkerAgent : MujocoAgent {

    public override void AgentReset()
    {
        base.AgentReset();

        // set to true this to show monitor while training
        Monitor.SetActive(true);

        StepRewardFunction = StepRewardWalker106;
        TerminateFunction = TerminateOnNonFootHitTerrain;
        ObservationsFunction = ObservationsDefault;

        BodyParts["pelvis"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="torso");
        BodyParts["left_thigh"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="left_thigh");
        BodyParts["right_thigh"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="right_thigh");
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
        
        AddVectorObs(SensorIsInTouch);
        JointRotations.ForEach(x=>AddVectorObs(x));
        AddVectorObs(JointVelocity);
    }

    float StepRewardWalker106()
    {
        float heightPenality = GetHeightPenality(1.1f);
        float uprightBonus = GetForwardBonus("pelvis");
        float velocity = GetVelocity();
        float effort = GetEffort();
        // var effortPenality = 1e-3f * (float)effort;
        var effortPenality = 1e-1f * (float)effort;

        var reward = velocity
            +uprightBonus
            -heightPenality
            -effortPenality;
        if (ShowMonitor) {
            var hist = new []{reward,velocity,uprightBonus,-heightPenality,-effortPenality}.ToList();
            Monitor.Log("rewardHist", hist.ToArray());
            // Monitor.Log("effortPenality", effortPenality, MonitorType.text);
            // Monitor.Log("reward", reward, MonitorType.text);
        }

        return reward;
    }
}
