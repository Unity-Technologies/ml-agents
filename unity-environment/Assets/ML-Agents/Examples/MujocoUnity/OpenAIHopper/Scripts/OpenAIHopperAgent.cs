using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MujocoUnity;
using UnityEngine;

public class OpenAIHopperAgent : MujocoAgent {

    public override void AgentReset()
    {
        base.AgentReset();

        // set to true this to show monitor while training
        Monitor.SetActive(true);

        StepRewardFunction = StepReward_Walker106;
        TerminateFunction = Terminate_Hopper;
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
        // AddVectorObs(MujocoController.FocalPointPosition);
        // AddVectorObs(MujocoController.FocalPointPositionVelocity); // acceleromoter (with out gravety)
        // AddVectorObs(MujocoController.FocalPointRotation);
        // AddVectorObs(MujocoController.FocalPointRotationVelocity);

        AddVectorObs(pelvis.velocity);
        AddVectorObs(pelvis.transform.forward); // gyroscope 
        AddVectorObs(pelvis.transform.up);
        // AddVectorObs(pelvis.angularVelocity); 
        // AddVectorObs(pelvis.rotation);
        
        AddVectorObs(MujocoController.SensorIsInTouch);
        MujocoController.JointRotations.ForEach(x=>AddVectorObs(x));
        MujocoController.JointAngularVelocities.ForEach(x=>AddVectorObs(x));
    }

    bool Terminate_Hopper()
    {
        if (Terminate_OnNonFootHitTerrain())
            return true;
        var height = GetHeightPenality(.3f);
        var angle = GetForwardBonus("pelvis");
        bool endOnHeight = height > 0f;
        bool endOnAngle = (angle < .4f);
        return endOnHeight || endOnAngle;        
    }

    float StepReward_Walker106()
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
            Monitor.Log("rewardHist", hist, MonitorType.hist);
        }

        return reward;
    }
}
