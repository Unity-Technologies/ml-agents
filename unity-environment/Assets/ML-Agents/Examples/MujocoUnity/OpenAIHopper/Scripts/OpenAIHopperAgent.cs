using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MujocoUnity;
using UnityEngine;
using MLAgents;

public class OpenAIHopperAgent : MujocoAgent {

    public override void AgentReset()
    {
        base.AgentReset();

        // set to true this to show monitor while training
        Monitor.SetActive(true);

        StepRewardFunction = StepRewardWalker106;
        TerminateFunction = TerminateHopper;
        ObservationsFunction = ObservationsDefault;

        BodyParts["pelvis"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="torso_geom");
        BodyParts["foot"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="foot_geom");
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
        var foot = BodyParts["foot"];
        AddVectorObs(foot.transform.position.y);
    }

    bool TerminateHopper()
    {
        if (TerminateOnNonFootHitTerrain())
            return true;
        var height = GetHeightPenality(.3f);
        var angle = GetForwardBonus("pelvis");
        bool endOnHeight = height > 0f;
        bool endOnAngle = (angle < .4f);
        return endOnHeight || endOnAngle;        
    }

    float StepRewardWalker106()
    {
        float heightPenality = GetHeightPenality(.7f);
        float uprightBonus = GetForwardBonus("pelvis");
        float velocity = GetVelocity();
        float effort = GetEffort();
        var effortPenality = 1e-1f * (float)effort;

        var reward = velocity
            +uprightBonus
            -heightPenality
            -effortPenality;
        if (ShowMonitor) {
            var hist = new []{reward,velocity,uprightBonus,-heightPenality,-effortPenality}.ToList();
            Monitor.Log("rewardHist", hist.ToArray());
        }

        return reward;
    }
}
