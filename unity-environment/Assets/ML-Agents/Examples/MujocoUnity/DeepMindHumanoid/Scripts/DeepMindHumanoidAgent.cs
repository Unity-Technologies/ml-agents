using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MujocoUnity;
using UnityEngine;

public class DeepMindHumanoidAgent : MujocoAgent {

    public override void AgentReset()
    {
        base.AgentReset();

        // set to true this to show monitor while training
        Monitor.SetActive(true);

        StepRewardFunction = StepRewardDeepMindHumanoid101;
        // StepRewardFunction = StepRewardOaiHumanoidRunOnSpot161;
        TerminateFunction = TerminateOnNonFootHitTerrain;
        ObservationsFunction = ObservationsHumanoid;

        BodyParts["head"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="head");
        BodyParts["shoulders"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="torso");
        BodyParts["waist"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="lower_waist");
        BodyParts["pelvis"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="butt");
        BodyParts["left_thigh"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="left_thigh");
        BodyParts["right_thigh"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="right_thigh");
        BodyParts["left_uarm"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="left_upper_arm");
        BodyParts["right_uarm"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="right_upper_arm");
        
        base.SetupBodyParts();

        // set up phase 
        PhaseBonusInitalize();
    }


    public override void AgentOnDone()
    {

    }
    void ObservationsHumanoid()
    {
        if (ShowMonitor) {
        }
        var pelvis = BodyParts["pelvis"];
        var shoulders = BodyParts["shoulders"];

        AddVectorObs(pelvis.velocity);
        AddVectorObs(pelvis.transform.forward); // gyroscope 
        AddVectorObs(pelvis.transform.up);
        
        AddVectorObs(shoulders.transform.forward); // gyroscope 
        AddVectorObs(shoulders.transform.up);

        AddVectorObs(MujocoController.SensorIsInTouch);
        MujocoController.JointRotations.ForEach(x=>AddVectorObs(x));
        AddVectorObs(MujocoController.JointVelocity);
    }



    float GetHumanoidArmEffort()
    {
        var mJoints = MujocoController.MujocoJoints
            .Where(x=>x.JointName.ToLowerInvariant().Contains("shoulder") || x.JointName.ToLowerInvariant().Contains("elbow"))
            .ToList();
        var effort = mJoints
            .Select(x=>Actions[MujocoController.MujocoJoints.IndexOf(x)])
            .Select(x=>Mathf.Pow(Mathf.Abs(x),2))
            .Sum();
        return effort;            
    }
    
    float StepRewardDeepMindHumanoid101()
    {
        float velocity = GetVelocity();
        float heightPenality = GetHeightPenality(1.2f);
        float uprightBonus = 
            (GetUprightBonus("shoulders") / 6)
            + (GetUprightBonus("waist") / 6)
            + (GetUprightBonus("pelvis") / 6);
        float forwardBonus = 
            (GetForwardBonus("shoulders")/ 4)
            + (GetForwardBonus("waist") / 6)
            + (GetForwardBonus("pelvis") / 6);
        
        float leftThighPenality = Mathf.Abs(GetLeftBonus("left_thigh"));
        float rightThighPenality = Mathf.Abs(GetRightBonus("right_thigh"));
        // float leftUarmPenality = Mathf.Abs(GetLeftBonus("left_uarm"));
        // float rightUarmPenality = Mathf.Abs(GetRightBonus("right_uarm"));
        // float limbPenalty = leftThighPenality + rightThighPenality + leftUarmPenality + rightUarmPenality;
        float limbPenalty = leftThighPenality + rightThighPenality;
        limbPenalty = Mathf.Min(0.5f, limbPenalty);
        float phaseBonus = GetPhaseBonus();
        var jointsAtLimitPenality = GetJointsAtLimitPenality() * 4;
        float effort = GetEffort(new string []{"right_hip_y", "right_knee", "left_hip_y", "left_knee"});
        var effortPenality = 0.05f * (float)effort;
        var reward = velocity 
            + uprightBonus
            + forwardBonus
            + phaseBonus
            - heightPenality
            - limbPenalty
            - jointsAtLimitPenality
            - effortPenality;
        if (ShowMonitor) {
            // var hist = new []{reward,velocity, shouldersUprightBonus, pelvisUprightBonus, headForwardBonus,- heightPenality,-effortPenality}.ToList();
            var hist = new []{
                reward, velocity, 
                uprightBonus, 
                forwardBonus, 
                phaseBonus, 
                -heightPenality, 
                -limbPenalty, 
                -jointsAtLimitPenality, 
                -effortPenality}.ToList();
            Monitor.Log("rewardHist", hist, MonitorType.hist);
        }
        return reward;            
    }
    float StepRewardOaiHumanoidRunOnSpot161()
    {
        float velocity = GetVelocity();
        float heightPenality = GetHeightPenality(1.2f);
        float uprightBonus = 
            (GetUprightBonus("shoulders") / 6)
            + (GetUprightBonus("waist") / 6)
            + (GetUprightBonus("pelvis") / 6);
        float forwardBonus = 
            (GetForwardBonus("shoulders")/ 2)
            + (GetForwardBonus("waist") / 6)
            + (GetForwardBonus("pelvis") / 6);
        
        float leftThighPenality = Mathf.Abs(GetLeftBonus("left_thigh"));
        float rightThighPenality = Mathf.Abs(GetRightBonus("right_thigh"));
        // float leftUarmPenality = Mathf.Abs(GetLeftBonus("left_uarm"));
        // float rightUarmPenality = Mathf.Abs(GetRightBonus("right_uarm"));
        // float limbPenalty = leftThighPenality + rightThighPenality + leftUarmPenality + rightUarmPenality;
        float limbPenalty = leftThighPenality + rightThighPenality;
        limbPenalty = Mathf.Min(0.5f, limbPenalty);
        float phaseBonus = GetPhaseBonus() * 5;
        var jointsAtLimitPenality = GetJointsAtLimitPenality() * 4;
        float effort = GetEffort(new string []{"right_hip_y", "right_knee", "left_hip_y", "left_knee"});
        var effortPenality = 0.5f * (float)effort;
        var reward = 0
            // + velocity 
            + uprightBonus
            + forwardBonus
            + phaseBonus
            - heightPenality
            - limbPenalty
            - jointsAtLimitPenality
            - effortPenality;
        if (ShowMonitor) {
            // var hist = new []{reward,velocity, shouldersUprightBonus, pelvisUprightBonus, headForwardBonus,- heightPenality,-effortPenality}.ToList();
            var hist = new []{
                reward, 
                // velocity, 
                uprightBonus, 
                forwardBonus, 
                phaseBonus, 
                -heightPenality, 
                -limbPenalty, 
                -jointsAtLimitPenality, 
                -effortPenality}.ToList();
            Monitor.Log("rewardHist", hist, MonitorType.hist);
        }
        return reward;            
    }

    // implement phase bonus (reward for left then right)
    List<float> _lastSenorState;
    float _phaseBonus;
    int _phase;

    public float LeftMin;
    public float LeftMax;

    public float RightMin;
    public float RightMax;

    void PhaseBonusInitalize()
    {
        _lastSenorState = Enumerable.Repeat<float>(0f, NumSensors).ToList();
        _phase = 0;
        _phaseBonus = 0f;
        PhaseResetLeft();
        PhaseResetRight();
    }

    void PhaseResetLeft()
    {
        LeftMin = float.MaxValue;
        LeftMax = float.MinValue;
        PhaseSetLeft();
    }
    void PhaseResetRight()
    {
        RightMin = float.MaxValue;
        RightMax = float.MinValue;
        PhaseSetRight();
    }
    void PhaseSetLeft()
    {
        var inPhaseToFocalAngle = BodyPartsToFocalRoation["left_thigh"] * BodyParts["left_thigh"].transform.right;
        var inPhaseAngleFromUp = Vector3.Angle(inPhaseToFocalAngle, Vector3.up);

        var angle = 180 - inPhaseAngleFromUp;
        var qpos2 = (angle % 180 ) / 180;
        var bonus = 2 - (Mathf.Abs(qpos2)*2)-1;
        LeftMin = Mathf.Min(LeftMin, bonus);
        LeftMax = Mathf.Max(LeftMax, bonus);
    }
    void PhaseSetRight()
    {
        var inPhaseToFocalAngle = BodyPartsToFocalRoation["right_thigh"] * BodyParts["right_thigh"].transform.right;
        var inPhaseAngleFromUp = Vector3.Angle(inPhaseToFocalAngle, Vector3.up);

        var angle = 180 - inPhaseAngleFromUp;
        var qpos2 = (angle % 180 ) / 180;
        var bonus = 2 - (Mathf.Abs(qpos2)*2)-1;
        RightMin = Mathf.Min(RightMin, bonus);
        RightMax = Mathf.Max(RightMax, bonus);
    }
    float CalcPhaseBonus(float min, float max)
    {
        float bonus = 0f;
        if (min < 0f && max < 0f) {
            min = Mathf.Abs(min);
            max = Mathf.Abs(max);
        } else if (min < 0f) {
            bonus = Mathf.Abs(min);
            min = 0f;
        }
        bonus += max-min;
        return bonus;
    }

    float GetPhaseBonus()
    {
        bool noPhaseChange = true;
        for (int i = 0; i < MujocoController.SensorIsInTouch.Count; i++)
        {
            noPhaseChange = noPhaseChange && MujocoController.SensorIsInTouch[i] == _lastSenorState[i];
            _lastSenorState[i] = MujocoController.SensorIsInTouch[i];
        }
        // special case: both feet in air
        if (MujocoController.SensorIsInTouch.Sum() == 0f)
            noPhaseChange = true;
        // special case: both feed down
        if (MujocoController.SensorIsInTouch.Sum() == 2f) {
            _phaseBonus = 0f;
            PhaseResetLeft();
            PhaseResetRight();
            return _phaseBonus;
        }
        
        if (noPhaseChange){
            // check if this is next best angle
            PhaseSetLeft();
            PhaseSetRight();
            var bonus = _phaseBonus;
            _phaseBonus *= 0.9f;
            return bonus;
        }

        // new phase
        _phaseBonus = 0;
        bool isLeftFootDown = MujocoController.SensorIsInTouch[0] != 0f;
        if (_phase == 2 && isLeftFootDown) {
            _phaseBonus = CalcPhaseBonus(LeftMin, LeftMax);
            _phaseBonus += 0.1f;
            PhaseResetLeft();
        } else if (_phase == 1 && !isLeftFootDown) {
            _phaseBonus = CalcPhaseBonus(RightMin, RightMax);
            _phaseBonus += 0.1f;
            PhaseResetRight();
        }
        _phase = isLeftFootDown ? 1 : 2;
        return _phaseBonus;
    }    
}
