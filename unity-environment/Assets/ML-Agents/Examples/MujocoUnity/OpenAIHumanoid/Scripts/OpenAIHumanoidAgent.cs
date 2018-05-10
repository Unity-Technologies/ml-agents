using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MujocoUnity;
using UnityEngine;

public class OpenAIHumanoidAgent : MujocoAgent {



    public override void CollectObservations()
    {
        _mujocoController.UpdateQFromExternalComponent();
        _observations();
    }

    public override void AgentAction(float[] vectorAction, string textAction)
	{
        Actions = vectorAction
            .Select(x=>Mathf.Clamp(x, -3, 3f)/3)
            .ToList();
        //KillJointPower(new []{"shoulder", "elbow"}); // HACK
        // if (ShowMonitor)
        //     Monitor.Log("actions", _actions, MonitorType.hist);
        for (int i = 0; i < _mujocoController.MujocoJoints.Count; i++) {
            var inp = (float)Actions[i];
            MujocoController.ApplyAction(_mujocoController.MujocoJoints[i], inp);
        }
        _mujocoController.UpdateFromExternalComponent();
        
        var done = _terminate();

        if (done)
        {
            Done();
            var reward = -1f;
            SetReward(reward);
        }
        if (!IsDone())
        {
            var reward = _stepReward();
            SetReward(reward);
        }
        base.AgentAction(vectorAction, textAction);
    }

    public override void AgentReset()
    {
            Monitor.SetActive(true);
            _mujocoController = GetComponent<MujocoController>();
            _mujocoController.MujocoJoints = null;
            _mujocoController.MujocoSensors = null;
            // var joints = this.GetComponentsInChildren<Joint>().ToList();
            // foreach (var item in joints)
            //     Destroy(item.gameObject);
            var rbs = this.GetComponentsInChildren<Rigidbody>().ToList();
            foreach (var item in rbs){
                if (item != null) 
                    DestroyImmediate(item.gameObject);
            }
            Resources.UnloadUnusedAssets();

            var mujocoSpawner = this.GetComponent<MujocoUnity.MujocoSpawner>();
            // if (mujocoSpawner != null)
                // mujocoSpawner.MujocoXml = MujocoXml;
            mujocoSpawner.SpawnFromXml();
            SetupMujoco();
            _mujocoController.UpdateFromExternalComponent();
            // switch(ActorId)
            // {
            //     case "a_oai_walker2d-v0":
            //         _stepReward = StepReward_OaiWalker;
            //         _terminate = Terminate_OnNonFootHitTerrain;
            //         _observations = Observations_Default;
            //         break;
            //     case "a_dm_walker-v0":
            //         _stepReward = StepReward_DmWalker;
            //         _terminate = Terminate_OnNonFootHitTerrain;
            //         _observations = Observations_Default;
            //         break;
            //     case "a_oai_hopper-v0":
            //         _stepReward = StepReward_OaiHopper;
            //         _terminate = Terminate_HopperOai;
            //         _observations = Observations_Default;
            //         break;
            //     case "a_oai_humanoid-v0":
                    _stepReward = StepReward_OaiHumanoidRun134;
                    //_stepReward = StepReward_OaiHumanoidRun;
                    // _stepReward = StepReward_OaiHumanoidStand;
                    // _stepReward = StepReward_OaiHumanoidPureRun;
                    _terminate = Terminate_OnNonFootHitTerrain;
                    _observations = Observations_Humanoid;
                    _bodyParts["pelvis"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="butt");
                    _bodyParts["shoulders"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="torso1");
                    _bodyParts["head"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="head");
                    _bodyParts["left_thigh"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="left_thigh1");
                    _bodyParts["right_thigh"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="right_thigh1");
                    _bodyParts["left_uarm"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="left_uarm1");
                    _bodyParts["right_uarm"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="right_uarm1");
            //         break;
            //     case "a_ant-v0":
            //     case "a_oai_half_cheetah-v0":
            //     default:
            //         throw new NotImplementedException();
            // }
            // set body part directions
            foreach (var bodyPart in _bodyParts)
            {
                var name = bodyPart.Key;
                var rigidbody = bodyPart.Value;

                // find up
                var focalPoint = rigidbody.position;
                focalPoint.x += 10;
                var focalPointRotation = rigidbody.rotation;
                focalPointRotation.SetLookRotation(focalPoint - rigidbody.position);
                _bodyPartsToFocalRoation[name] = focalPointRotation;
            }
    }

    public override void AgentOnDone()
    {

    }
        void Observations_Humanoid()
        {
            if (ShowMonitor) {
                //Monitor.Log("pos", _mujocoController.qpos, MonitorType.hist);
                //Monitor.Log("vel", _mujocoController.qvel, MonitorType.hist);
                //Monitor.Log("onSensor", _mujocoController.OnSensor, MonitorType.hist);
                //Monitor.Log("sensor", _mujocoController.SensorIsInTouch, MonitorType.hist);
            }
            var pelvis = _bodyParts["pelvis"];
            var shoulders = _bodyParts["shoulders"];
            AddVectorObs(_mujocoController.FocalPointPosition);
            AddVectorObs(_mujocoController.FocalPointPositionVelocity); // acceleromoter (with out gravety)
            AddVectorObs(_mujocoController.FocalPointRotation);
            AddVectorObs(_mujocoController.FocalPointRotationVelocity);

            // var focalTransform = _focalPoint.transform;
            // var focalRidgedBody = _focalPoint.GetComponent<Rigidbody>();
            // FocalPointPosition = focalTransform.position;
            // FocalPointPositionVelocity = focalRidgedBody.velocity;
            // var lastFocalPointRotationVelocity = FocalPointRotation;
            // FocalPointEulerAngles = focalTransform.eulerAngles;
            // FocalPointRotation = new Vector3(
            //     ((FocalPointEulerAngles.x - 180f) % 180 ) / 180,
            //     ((FocalPointEulerAngles.y - 180f) % 180 ) / 180,
            //     ((FocalPointEulerAngles.z - 180f) % 180 ) / 180);
            // FocalPointRotationVelocity = FocalPointRotation-lastFocalPointRotationVelocity;
            AddVectorObs(pelvis.velocity);
            AddVectorObs(pelvis.transform.forward); // gyroscope 
            AddVectorObs(pelvis.transform.up);
            AddVectorObs(pelvis.angularVelocity); 
            AddVectorObs(pelvis.rotation);
            
            AddVectorObs(shoulders.transform.forward); // gyroscope 

            AddVectorObs(_mujocoController.SensorIsInTouch);
            //AddVectorObs(_mujocoController.JointAngles);
            //AddVectorObs(_mujocoController.JointVelocity);
            _mujocoController.JointRotations.ForEach(x=>AddVectorObs(x));
            AddVectorObs(_mujocoController.JointVelocity);
        }

        public float[] Low;
		public float[] High;
        public bool ShowMonitor;
		float[] _observation1D;
        float[] _internalLow;
        float[] _internalHigh;
        int _jointSize = 13; // 9+4
        int _numJoints = 3; // for debug object
        int _sensorOffset; // offset in observations to where senors begin
        int _numSensors;
        int _sensorSize; // number of floats per senor
        int _observationSize; // total number of floats
        

        MujocoController _mujocoController;

        public List<float> Actions;
        Func<bool> _terminate;
        Func<float> _stepReward;
        Action _observations;
        Dictionary<string,Rigidbody> _bodyParts = new Dictionary<string,Rigidbody>();
        Dictionary<string,Quaternion> _bodyPartsToFocalRoation = new Dictionary<string,Quaternion>();

        int _frameSkip = 5; // number of physics frames to skip between training
        int _nSteps = 1000; // total number of training steps
        List<float> _lastSenorState;
        float _nextPhaseBonus;
        float _phaseBonus;
        int _phase;
        public void SetupMujoco()
        {
            _mujocoController = GetComponent<MujocoController>();
            _numJoints = _mujocoController.qpos.Count;
            _numSensors = _mujocoController.MujocoSensors.Count;            
            _jointSize = 2;
            _sensorSize = 1;
            _sensorOffset = _jointSize * _numJoints;
            _observationSize = _sensorOffset + (_sensorSize * _numSensors);
            _observation1D = Enumerable.Repeat<float>(0f, _observationSize).ToArray();
            Low = _internalLow = Enumerable.Repeat<float>(float.MinValue, _observationSize).ToArray();
            High = _internalHigh = Enumerable.Repeat<float>(float.MaxValue, _observationSize).ToArray();
            for (int j = 0; j < _numJoints; j++)
            {
                var offset = j * _jointSize;
                _internalLow[offset+0] = -5;//-10;
                _internalHigh[offset+0] = 5;//10;
                _internalLow[offset+1] = -5;//-500;
                _internalHigh[offset+1] = 5;//500;
                // _internalLow[offset+2] = -5;//-500;
                // _internalHigh[offset+3] = 5;//500;
            }
            for (int j = 0; j < _numSensors; j++)
            {
                var offset = _sensorOffset + (j * _sensorSize);
                _internalLow[offset+0] = -1;//-10;
                _internalHigh[offset+0] = 1;//10;
            }    
            _lastSenorState = Enumerable.Repeat<float>(0f, _numSensors).ToList();
            _phase = 0;
            _phaseBonus = 0f;
            _nextPhaseBonus = 0f;
            //this.brain = GetComponent<Brain>();
        }

		float StepReward_OaiWalker()
		{
			return StepReward_DmWalker();
		}
        float GetHeight()
        {
			var feetYpos = _mujocoController.MujocoJoints
				.Where(x=>x.JointName.ToLowerInvariant().Contains("foot"))
				.Select(x=>x.Joint.transform.position.y)
				.OrderBy(x=>x)
				.ToList();
            float lowestFoot = 0f;
            if(feetYpos!=null && feetYpos.Count != 0)
                lowestFoot = feetYpos[0];
			var height = _mujocoController.FocalPointPosition.y - lowestFoot;
            return height;
        }
        float GetVelocity()
        {
			var dt = Time.fixedDeltaTime;
			var rawVelocity = _mujocoController.FocalPointPositionVelocity.x;
            var maxSpeed = 4f; // meters per second
            //rawVelocity = Mathf.Clamp(rawVelocity,-maxSpeed,maxSpeed);
			var velocity = rawVelocity / maxSpeed;
            if (ShowMonitor) {
                Monitor.Log("MPH: ", rawVelocity * 2.236936f, MonitorType.text);
                // Monitor.Log("rawVelocity", rawVelocity, MonitorType.text);
                // Monitor.Log("velocity", velocity, MonitorType.text);
            }
            return velocity;
        }
        float GetUprightBonus()
        {
            var qpos2 = (GetAngleFromUp() % 180 ) / 180;
            var uprightBonus = 0.5f * (2 - (Mathf.Abs(qpos2)*2)-1);
            // if (ShowMonitor)
                // Monitor.Log("uprightBonus", uprightBonus, MonitorType.text);
            return uprightBonus;
        }
        float GetUprightBonus(string bodyPart)
        {
            var toFocalAngle = _bodyPartsToFocalRoation[bodyPart] * -_bodyParts[bodyPart].transform.forward;
            var angleFromUp = Vector3.Angle(toFocalAngle, Vector3.up);
            var qpos2 = (angleFromUp % 180 ) / 180;
            var uprightBonus = 0.5f * (2 - (Mathf.Abs(qpos2)*2)-1);
            // if (ShowMonitor)
            //     Monitor.Log($"upright[{bodyPart}] Bonus", uprightBonus, MonitorType.text);
            return uprightBonus;
        }

        float GetDirectionBonus(string bodyPart, Vector3 direction, float maxBonus = 0.5f)
        {
            var toFocalAngle = _bodyPartsToFocalRoation[bodyPart] * _bodyParts[bodyPart].transform.right;
            var angle = Vector3.Angle(toFocalAngle, direction);
            var qpos2 = (angle % 180 ) / 180;
            var bonus = maxBonus * (2 - (Mathf.Abs(qpos2)*2)-1);
            return bonus;
        }
        void GetDirectionDebug(string bodyPart)
        {
            var toFocalAngle = _bodyPartsToFocalRoation[bodyPart] * _bodyParts[bodyPart].transform.right;
            var angleFromLeft = Vector3.Angle(toFocalAngle, Vector3.left);
            var angleFromUp = Vector3.Angle(toFocalAngle, Vector3.up);
            var angleFromDown = Vector3.Angle(toFocalAngle, Vector3.down);
            var angleFromRight = Vector3.Angle(toFocalAngle, Vector3.right);
            var angleFromForward = Vector3.Angle(toFocalAngle, Vector3.forward);
            var angleFromBack = Vector3.Angle(toFocalAngle, Vector3.back);
            print ($"{bodyPart}: l: {angleFromLeft}, r: {angleFromRight}, f: {angleFromForward}, b: {angleFromBack}, u: {angleFromUp}, d: {angleFromDown}");
        }

        float GetLeftBonus(string bodyPart)
        {
            var bonus = GetDirectionBonus(bodyPart, Vector3.left);
            // if (ShowMonitor)
            //     Monitor.Log($"left[{bodyPart}] Bonus", bonus, MonitorType.text);
            // print (bonus);
            return bonus;
        }       
        float GetRightBonus(string bodyPart)
        {
            var bonus = GetDirectionBonus(bodyPart, Vector3.right);
            // if (ShowMonitor)
            //     Monitor.Log($"right[{bodyPart}] Bonus", bonus, MonitorType.text);
            // print (bonus);
            return bonus;
        }       
        float GetForwardBonus(string bodyPart)
        {
            var bonus = GetDirectionBonus(bodyPart, Vector3.forward);
            // if (ShowMonitor)
            //     Monitor.Log($"forward[{bodyPart}] Bonus", bonus, MonitorType.text);
            // print (bonus);
            return bonus;
        }
        float GetHeightPenality(float maxHeight)
        {
            var height = GetHeight();
            var heightPenality = maxHeight - height;
			heightPenality = Mathf.Clamp(heightPenality, 0f, maxHeight);
            // if (ShowMonitor) {
            //     Monitor.Log("height", height, MonitorType.text);
            //     Monitor.Log("heightPenality", heightPenality, MonitorType.text);
            // }
            return heightPenality;
        }
        void KillJointPower(string[] hints)
        {
            var mJoints = hints
                .SelectMany(hint=>
                    _mujocoController.MujocoJoints
                        .Where(x=>x.JointName.ToLowerInvariant().Contains(hint.ToLowerInvariant()))
                ).ToList();
            foreach (var joint in mJoints)
                Actions[_mujocoController.MujocoJoints.IndexOf(joint)] = 0f;
        }
        float GetHumanoidArmEffort()
        {
            var mJoints = _mujocoController.MujocoJoints
                .Where(x=>x.JointName.ToLowerInvariant().Contains("shoulder") || x.JointName.ToLowerInvariant().Contains("elbow"))
                .ToList();
            var effort = mJoints
                .Select(x=>Actions[_mujocoController.MujocoJoints.IndexOf(x)])
				.Select(x=>Mathf.Pow(Mathf.Abs(x),2))
				.Sum();
            return effort;            
        }
        float GetEffort(string[] ignorJoints = null)
        {
            double effort = 0;
            for (int i = 0; i < Actions.Count; i++)
            {
                var name = _mujocoController.MujocoJoints[i].JointName;
                if (ignorJoints != null && ignorJoints.Contains(name))
                    continue;
                var jointEffort = Mathf.Pow(Mathf.Abs(Actions[i]),2);
                effort += jointEffort;
            }
            return (float)effort;
			// var effort = _actions
			// 	.Select(x=>Mathf.Pow(Mathf.Abs(x),2))
			// 	.Sum();
            // // if (ShowMonitor)
            //     // Monitor.Log("effort", effort, MonitorType.text);
            // return effort;
        }
        float GetJointsAtLimitPenality(string[] ignorJoints = null)
        {
            int atLimitCount = 0;
            for (int i = 0; i < Actions.Count; i++)
            {
                var name = _mujocoController.MujocoJoints[i].JointName;
                if (ignorJoints != null && ignorJoints.Contains(name))
                    continue;
                bool atLimit = Mathf.Abs(Actions[i]) >= 1f;
                if (atLimit)
                    atLimitCount++;
            }
            float penality = atLimitCount * 0.2f;
            return (float)penality;            
        }
        float GetEffortSum()
        {
			var effort = Actions
				.Select(x=>Mathf.Abs(x))
				.Sum();
            return effort;
        }
        float GetEffortMean()
        {
			var effort = Actions
				.Average();
            return effort;
        }

        float GetAngleFromUp()
        {
            var angleFromUp = Vector3.Angle(_mujocoController._focalPoint.transform.forward, Vector3.up);
            if (ShowMonitor) {
                Monitor.Log("AngleFromUp", angleFromUp);
            }
            return angleFromUp; 
        }
		float StepReward_DmWalker()
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
        
        float GymHumanoidReward()
        {
            // alive_bonus = 5.0
            // data = self.sim.data
            // lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
            // quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
            // quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
            // quad_impact_cost = min(quad_impact_cost, 10)
            // reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus            
            var alive_bonus = 5f;
            var lin_vel_cost = 0.25f * GetVelocity();
            float quad_ctrl_cost = 0.1f * GetEffort();
            float quad_impact_cost = 0; // .5e-6 * np.square(data.cfrc_ext).sum() // force on body   
            // quad_impact_cost = min(quad_impact_cost, 10)
            var reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus;
            if (ShowMonitor) {
                var hist = new []{reward,lin_vel_cost, -quad_ctrl_cost, -quad_impact_cost, alive_bonus}.ToList();
                Monitor.Log("rewardHist", hist, MonitorType.hist);
                //Monitor.Log("effortPenality", effortPenality, MonitorType.text);
            }
            return reward;
        }
        float StepReward_OaiHumanoidStand()
        {
            float heightPenality = GetHeightPenality(1.2f);
            float shouldersUprightBonus = GetUprightBonus("shoulders") / 2;
            float pelvisUprightBonus = GetUprightBonus("pelvis") / 2;
            float effort = GetEffort();
            var effortPenality = 0.02f * (float)effort;
            var armPenalty = 0.1f * (float)GetHumanoidArmEffort();
			var reward = 0f 
                + shouldersUprightBonus
                + pelvisUprightBonus
                - heightPenality
			    - effortPenality
                - armPenalty;
            if (ShowMonitor) {
                var hist = new []{reward, shouldersUprightBonus, pelvisUprightBonus,- heightPenality,-effortPenality, -armPenalty}.ToList();
                Monitor.Log("rewardHist", hist, MonitorType.hist);
            }
			return reward;      
        }

        float StepReward_OaiHumanoidBasicRun()
        {
            float velocity = GetVelocity();
            float heightPenality = GetHeightPenality(1.2f);
            float shouldersUprightBonus = GetUprightBonus("shoulders") / 4;
            float pelvisUprightBonus = GetUprightBonus("pelvis") / 4;
            // float headForwardBonus = GetForwardBonus("shoulders") / 2;
            float headForwardBonus = GetForwardBonus("head") / 4;
            float pelvisForwardBonus = GetForwardBonus("pelvis") / 4;
            
            // float leftThighPenality = Mathf.Abs(GetLeftBonus("left_thigh"));
            // float rightThighPenality = Mathf.Abs(GetRightBonus("right_thigh"));
            // float leftUarmPenality = Mathf.Abs(GetLeftBonus("left_uarm"));
            // float rightUarmPenality = Mathf.Abs(GetRightBonus("right_uarm"));
            // float limbPenalty = leftThighPenality + rightThighPenality + leftUarmPenality + rightUarmPenality;
            // limbPenalty = Mathf.Min(0.5f, limbPenalty);
            // // GetDirectionDebug("left_thigh");
            // GetDirectionDebug("right_thigh");
            // print (ProcessLegPhaseStep("right_thigh", "left_thigh"));
            // float phaseBonus = GetPhaseBonus();
            var jointsAtLimitPenality = GetJointsAtLimitPenality();
            float effort = GetEffort(new string []{"right_hip_y", "right_knee", "left_hip_y", "left_knee"});
            var effortPenality = 0.5f * (float)effort;
			var reward = velocity 
                + shouldersUprightBonus
                + pelvisUprightBonus
                + headForwardBonus
                + pelvisForwardBonus
                // + phaseBonus
                - heightPenality
                // - limbPenalty
                - jointsAtLimitPenality
			    - effortPenality;
                // - armPenalty;
            if (ShowMonitor) {
                // var hist = new []{reward,velocity, shouldersUprightBonus, pelvisUprightBonus, headForwardBonus,- heightPenality,-effortPenality}.ToList();
                var hist = new []{
                    reward, velocity, shouldersUprightBonus, pelvisUprightBonus, 
                    headForwardBonus, pelvisForwardBonus, //phaseBonus, 
                    -heightPenality, -jointsAtLimitPenality, -effortPenality}.ToList();
                Monitor.Log("rewardHist", hist, MonitorType.hist);
            }
			return reward;            
        }
        
        float StepReward_OaiHumanoidRun()
        {
            float velocity = GetVelocity();
            float heightPenality = GetHeightPenality(1.2f);
            float shouldersUprightBonus = GetUprightBonus("shoulders") / 2;
            float pelvisUprightBonus = GetUprightBonus("pelvis") / 2;
            // float headForwardBonus = GetForwardBonus("shoulders") / 2;
            float headForwardBonus = GetForwardBonus("head") / 2;
            float pelvisForwardBonus = GetForwardBonus("pelvis") / 2;
            
            float leftThighPenality = Mathf.Abs(GetLeftBonus("left_thigh"));
            float rightThighPenality = Mathf.Abs(GetRightBonus("right_thigh"));
            float leftUarmPenality = Mathf.Abs(GetLeftBonus("left_uarm"));
            float rightUarmPenality = Mathf.Abs(GetRightBonus("right_uarm"));
            float limbPenalty = leftThighPenality + rightThighPenality + leftUarmPenality + rightUarmPenality;
            limbPenalty = Mathf.Min(0.5f, limbPenalty);
            // GetDirectionDebug("left_thigh");
            // GetDirectionDebug("right_thigh");
            // print (ProcessLegPhaseStep("right_thigh", "left_thigh"));
            float phaseBonus = GetPhaseBonus();
            var jointsAtLimitPenality = GetJointsAtLimitPenality();
            float effort = GetEffort(new string []{"right_hip_y", "right_knee", "left_hip_y", "left_knee"});
            var effortPenality = 0.5f * (float)effort;
			var reward = velocity 
                + shouldersUprightBonus
                + pelvisUprightBonus
                + headForwardBonus
                + pelvisForwardBonus
                + phaseBonus
                - heightPenality
                - limbPenalty
                - jointsAtLimitPenality
			    - effortPenality;
                // - armPenalty;
            if (ShowMonitor) {
                // var hist = new []{reward,velocity, shouldersUprightBonus, pelvisUprightBonus, headForwardBonus,- heightPenality,-effortPenality}.ToList();
                var hist = new []{
                    reward, velocity, shouldersUprightBonus, pelvisUprightBonus, 
                    headForwardBonus, pelvisForwardBonus, phaseBonus, 
                    -heightPenality, -limbPenalty, -jointsAtLimitPenality, -effortPenality}.ToList();
                Monitor.Log("rewardHist", hist, MonitorType.hist);
            }
			return reward;            
        }

        float StepReward_OaiHumanoidRun132()
        {
            float velocity = GetVelocity();
            float heightPenality = GetHeightPenality(1.2f);
            float shouldersUprightBonus = GetUprightBonus("shoulders") / 2;
            float pelvisUprightBonus = GetUprightBonus("pelvis") / 2;
            // float headForwardBonus = GetForwardBonus("shoulders") / 2;
            float headForwardBonus = GetForwardBonus("head") / 2;
            float pelvisForwardBonus = GetForwardBonus("pelvis") / 2;
            
            float leftThighPenality = Mathf.Abs(GetLeftBonus("left_thigh"));
            float rightThighPenality = Mathf.Abs(GetRightBonus("right_thigh"));
            float leftUarmPenality = Mathf.Abs(GetLeftBonus("left_uarm"));
            float rightUarmPenality = Mathf.Abs(GetRightBonus("right_uarm"));
            float limbPenalty = leftThighPenality + rightThighPenality + leftUarmPenality + rightUarmPenality;
            limbPenalty = Mathf.Min(0.5f, limbPenalty);
            // GetDirectionDebug("left_thigh");
            // GetDirectionDebug("right_thigh");
            // print (ProcessLegPhaseStep("right_thigh", "left_thigh"));
            float phaseBonus = GetPhaseBonus();
            var jointsAtLimitPenality = GetJointsAtLimitPenality() * 4;
            float effort = GetEffort(new string []{"right_hip_y", "right_knee", "left_hip_y", "left_knee"});
            var effortPenality = 0.5f * (float)effort;
			var reward = velocity 
                + shouldersUprightBonus
                + pelvisUprightBonus
                + headForwardBonus
                + pelvisForwardBonus
                + phaseBonus
                - heightPenality
                - limbPenalty
                - jointsAtLimitPenality
			    - effortPenality;
                // - armPenalty;
            if (ShowMonitor) {
                // var hist = new []{reward,velocity, shouldersUprightBonus, pelvisUprightBonus, headForwardBonus,- heightPenality,-effortPenality}.ToList();
                var hist = new []{
                    reward, velocity, shouldersUprightBonus, pelvisUprightBonus, 
                    headForwardBonus, pelvisForwardBonus, phaseBonus, 
                    -heightPenality, -limbPenalty, -jointsAtLimitPenality, -effortPenality}.ToList();
                Monitor.Log("rewardHist", hist, MonitorType.hist);
            }
			return reward;            
        }
        float StepReward_OaiHumanoidRun133()
        {
            float velocity = GetVelocity();
            float heightPenality = GetHeightPenality(1.2f);
            // float shouldersUprightBonus = GetUprightBonus("shoulders") / 2;
            // float pelvisUprightBonus = GetUprightBonus("pelvis") / 2;
            // // float headForwardBonus = GetForwardBonus("shoulders") / 2;
            // float headForwardBonus = GetForwardBonus("head") / 2;
            // float pelvisForwardBonus = GetForwardBonus("pelvis") / 2;
            
            // float leftThighPenality = Mathf.Abs(GetLeftBonus("left_thigh"));
            // float rightThighPenality = Mathf.Abs(GetRightBonus("right_thigh"));
            // float leftUarmPenality = Mathf.Abs(GetLeftBonus("left_uarm"));
            // float rightUarmPenality = Mathf.Abs(GetRightBonus("right_uarm"));
            // float limbPenalty = leftThighPenality + rightThighPenality + leftUarmPenality + rightUarmPenality;
            // limbPenalty = Mathf.Min(0.5f, limbPenalty);
            // // GetDirectionDebug("left_thigh");
            // // GetDirectionDebug("right_thigh");
            // // print (ProcessLegPhaseStep("right_thigh", "left_thigh"));
            // float phaseBonus = GetPhaseBonus();
            var jointsAtLimitPenality = GetJointsAtLimitPenality() * 4;
            float effort = GetEffort(new string []{"right_hip_y", "right_knee", "left_hip_y", "left_knee"});
            var effortPenality = 0.5f * (float)effort;
			var reward = velocity 
                // + shouldersUprightBonus
                // + pelvisUprightBonus
                // + headForwardBonus
                // + pelvisForwardBonus
                // + phaseBonus
                - heightPenality
                // - limbPenalty
                - jointsAtLimitPenality
			    - effortPenality;
                // - armPenalty;
            if (ShowMonitor) {
                // var hist = new []{reward,velocity, shouldersUprightBonus, pelvisUprightBonus, headForwardBonus,- heightPenality,-effortPenality}.ToList();
                var hist = new []{
                    reward, velocity, 
                    // shouldersUprightBonus, pelvisUprightBonus, 
                    // headForwardBonus, pelvisForwardBonus, phaseBonus, 
                    -heightPenality, 
                    // -limbPenalty, 
                    -jointsAtLimitPenality, 
                    -effortPenality}.ToList();
                Monitor.Log("rewardHist", hist, MonitorType.hist);
            }
			return reward;            
        }
        float StepReward_OaiHumanoidRun134()
        {
            float velocity = GetVelocity();
            float heightPenality = GetHeightPenality(1.2f);
            // float shouldersUprightBonus = GetUprightBonus("shoulders") / 2;
            float pelvisUprightBonus = GetUprightBonus("pelvis") / 2;
            // float headForwardBonus = GetForwardBonus("shoulders") / 2;
            // float headForwardBonus = GetForwardBonus("head") / 2;
            float pelvisForwardBonus = GetForwardBonus("pelvis") / 2;
            
            // float leftThighPenality = Mathf.Abs(GetLeftBonus("left_thigh"));
            // float rightThighPenality = Mathf.Abs(GetRightBonus("right_thigh"));
            // float leftUarmPenality = Mathf.Abs(GetLeftBonus("left_uarm"));
            // float rightUarmPenality = Mathf.Abs(GetRightBonus("right_uarm"));
            // float limbPenalty = leftThighPenality + rightThighPenality + leftUarmPenality + rightUarmPenality;
            // limbPenalty = Mathf.Min(0.5f, limbPenalty);
            // // GetDirectionDebug("left_thigh");
            // // GetDirectionDebug("right_thigh");
            // // print (ProcessLegPhaseStep("right_thigh", "left_thigh"));
            // float phaseBonus = GetPhaseBonus();
            var jointsAtLimitPenality = GetJointsAtLimitPenality() * 4;
            float effort = GetEffort(new string []{"right_hip_y", "right_knee", "left_hip_y", "left_knee"});
            var effortPenality = 0.5f * (float)effort;
			var reward = velocity 
                // + shouldersUprightBonus
                + pelvisUprightBonus
                // + headForwardBonus
                + pelvisForwardBonus
                // + phaseBonus
                - heightPenality
                // - limbPenalty
                - jointsAtLimitPenality
			    - effortPenality;
                // - armPenalty;
            if (ShowMonitor) {
                // var hist = new []{reward,velocity, shouldersUprightBonus, pelvisUprightBonus, headForwardBonus,- heightPenality,-effortPenality}.ToList();
                var hist = new []{
                    reward, velocity, 
                    // shouldersUprightBonus, pelvisUprightBonus, 
                    // headForwardBonus, pelvisForwardBonus, phaseBonus, 
                    -heightPenality, 
                    // -limbPenalty, 
                    -jointsAtLimitPenality, 
                    -effortPenality}.ToList();
                Monitor.Log("rewardHist", hist, MonitorType.hist);
            }
			return reward;            
        }
        float ProcessLegPhaseStep(string inPhaseLimb, string outPhaseLimb)
        {
            var inPhaseToFocalAngle = _bodyPartsToFocalRoation[inPhaseLimb] * _bodyParts[inPhaseLimb].transform.right;
            var inPhaseAngleFromUp = Vector3.Angle(inPhaseToFocalAngle, Vector3.up);

            var outPhaseToFocalAngle = _bodyPartsToFocalRoation[outPhaseLimb] * _bodyParts[outPhaseLimb].transform.right;
            var outAngleFromUp = Vector3.Angle(outPhaseToFocalAngle, Vector3.up);

            if (outAngleFromUp > 90f || inPhaseAngleFromUp < 90f)
                return 0f; // Leg is not back
            var maxBonus = .5f;
            var angle = 180 - inPhaseAngleFromUp;
            var qpos2 = (angle % 180 ) / 180;
            var bonus = maxBonus * (2 - (Mathf.Abs(qpos2)*2)-1);
            return bonus;
        }
        float GetPhaseBonus()
        {
            bool noPhaseChange = true;
            for (int i = 0; i < _mujocoController.SensorIsInTouch.Count; i++)
            {
                noPhaseChange = noPhaseChange && _mujocoController.SensorIsInTouch[i] == _lastSenorState[i];
                _lastSenorState[i] = _mujocoController.SensorIsInTouch[i];
            }
            // special case: no feed in air
            if (_mujocoController.SensorIsInTouch.Sum() == 0f)
                noPhaseChange = true;
            // special case: both feed down
            if (_mujocoController.SensorIsInTouch.Sum() == 2f) {
                _phaseBonus = 0f;
                _nextPhaseBonus = 0;
                return _phaseBonus;
            }
            
            if (noPhaseChange){
                // check if this is next best angle
                if (_phase == 1)
                    _nextPhaseBonus = Mathf.Max(_phaseBonus, ProcessLegPhaseStep("left_thigh", "right_thigh"));
                else if (_phase == 2)
                    _nextPhaseBonus = Mathf.Max(_phaseBonus, ProcessLegPhaseStep("right_thigh", "left_thigh"));
                var bonus = _phaseBonus;
                _phaseBonus *= 0.9f;
                return _phaseBonus;
            }

            // new phase
            bool invalidPhase = false;
            bool isLeftPhase = _mujocoController.SensorIsInTouch[0] != 0f;
            if (_phase == 1 && isLeftPhase)
                invalidPhase = true;
            else if (_phase == 2 && !isLeftPhase)
                invalidPhase = true;
            _phase = isLeftPhase ? 1 : 2;
            _phaseBonus = invalidPhase ? 0f : _nextPhaseBonus;
            _nextPhaseBonus = 0;
            return _phaseBonus;
        }

        float StepReward_OaiHumanoidPureRun()
        {
            float velocity = GetVelocity();
            float effort = GetEffort();
            var effortPenality = 0.1f * (float)effort;
			var reward = velocity 
			    - effortPenality;
            if (ShowMonitor) {
                var hist = new []{reward,velocity, -effortPenality}.ToList();
                Monitor.Log("rewardHist", hist, MonitorType.hist);
            }
			return reward;            
        }
        float StepReward_OaiHopper()
		{
            float heightPenality = GetHeightPenality(1.0f);
            float uprightBonus = GetUprightBonus();
            float velocity = GetVelocity();
            float effort = GetEffort();
			var alive_bonus = 0.5f;
            var effortPenality = 1e-3f * (float)effort;
            // var effortPenality = 1e-1f * (float)effort;
			var reward = velocity 
                + alive_bonus
			    - effortPenality;
            if (ShowMonitor) {
                var hist = new []{reward,velocity,alive_bonus,-effortPenality}.ToList();
                Monitor.Log("rewardHist", hist, MonitorType.hist);
                // Monitor.Log("effortPenality", effortPenality, MonitorType.text);
            }
			return reward;
		}      
		bool Terminate_Never()
		{
			return false;
		}
        bool Terminate_OnNonFootHitTerrain()
		{
			return NonFootHitTerrain;
		}
        bool Terminate_HopperOai()
		{
			if (NonFootHitTerrain)
				return true;
			if (_mujocoController.qpos == null)
				return false;
			var height = _mujocoController.qpos[1];
			var angle = Mathf.Abs(_mujocoController.qpos[2]);
			bool endOnHeight = (height < .3f);
			bool endOnAngle = (angle > (1f/180f) * (5.7296f *6));
			return endOnHeight || endOnAngle;
		}
}
