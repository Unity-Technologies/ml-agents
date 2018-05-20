using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MujocoUnity;
using UnityEngine;

public class DeepMindWalkerAgent : MujocoAgent {

    public override void AgentReset()
    {
        base.AgentReset();

        // set to true this to show monitor while training
        Monitor.SetActive(true);

        StepRewardFunction = StepReward_Walker106;
        TerminateFunction = Terminate_OnNonFootHitTerrain;
        ObservationsFunction = Observations_Default;

        BodyParts["pelvis"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="torso");
        BodyParts["left_thigh"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="left_thigh");
        BodyParts["right_thigh"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="right_thigh");
        base.SetupBodyParts();
    }


    public override void AgentOnDone()
    {

    }
    void Observations_Default()
    {
        if (ShowMonitor) {
            //Monitor.Log("onSensor", _mujocoController.OnSensor, MonitorType.hist);
            //Monitor.Log("sensor", _mujocoController.SensorIsInTouch, MonitorType.hist);
        }
        var pelvis = BodyParts["pelvis"];
        AddVectorObs(MujocoController.FocalPointPosition);
        AddVectorObs(MujocoController.FocalPointPositionVelocity); // acceleromoter (with out gravety)
        AddVectorObs(MujocoController.FocalPointRotation);
        AddVectorObs(MujocoController.FocalPointRotationVelocity);

        AddVectorObs(pelvis.velocity);
        AddVectorObs(pelvis.transform.forward); // gyroscope 
        AddVectorObs(pelvis.transform.up);
        AddVectorObs(pelvis.angularVelocity); 
        AddVectorObs(pelvis.rotation);
        
        // AddVectorObs(shoulders.transform.forward); // gyroscope 

        AddVectorObs(MujocoController.SensorIsInTouch);
        //AddVectorObs(_mujocoController.JointAngles);
        //AddVectorObs(_mujocoController.JointVelocity);
        MujocoController.JointRotations.ForEach(x=>AddVectorObs(x));
        AddVectorObs(MujocoController.JointVelocity);
    }

    float StepReward_Walker106()
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
            Monitor.Log("rewardHist", hist, MonitorType.hist);
            // Monitor.Log("effortPenality", effortPenality, MonitorType.text);
            // Monitor.Log("reward", reward, MonitorType.text);
        }

        return reward;
    }
}
