using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MujocoUnity;
using UnityEngine;

public class OpenAIWalker2dAgent : MujocoAgent {

    public override void AgentReset()
    {
        base.AgentReset();

        // set to true this to show monitor while training
        Monitor.SetActive(true);

        StepRewardFunction = StepReward_Walker101;
        TerminateFunction = Terminate_OnNonFootHitTerrain;
        ObservationsFunction = Observations_Default;

        BodyParts["pelvis"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="torso_geom");
        BodyParts["left_thigh"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="thigh_left_geom");
        BodyParts["right_thigh"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="thigh_geom");
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

    float StepReward_Walker101()
    {
        // float heightPenality = GetHeightPenality(1f);
        float heightPenality = GetHeightPenality(.65f);
        float uprightBonus = GetUprightBonus();
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
            Monitor.Log("rewardHist", hist, MonitorType.hist);
            // Monitor.Log("effortPenality", effortPenality, MonitorType.text);
            // Monitor.Log("reward", reward, MonitorType.text);
        }

        return reward;
    }
}
