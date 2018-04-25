using System.Collections.Generic;
using System.Linq;
using MujocoUnity;
using UnityEngine;
using System;

namespace MlaMujocoUnity {
    public class MujocoAgent : Agent 
    {
        Rigidbody rBody;
		public TextAsset MujocoXml;
        public string ActorId;
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
        bool _footHitTerrain;
        bool _nonFootHitTerrain;
        List<float> _actions;
        Func<bool> _terminate;
        Func<float> _stepReward;
        Action _observations;
        Dictionary<string,Rigidbody> _bodyParts = new Dictionary<string,Rigidbody>();

        int _frameSkip = 5; // number of physics frames to skip between training
        int _nSteps = 1000; // total number of training steps

        void Start () {
            rBody = GetComponent<Rigidbody>();
        }
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
            this.brain = GameObject.Find("MujocoBrain").GetComponent<Brain>();
        }

        public int GetObservationCount()
        {
            return _observation1D.Length;
        }
        public int GetActionsCount()
        {
            return _mujocoController.MujocoJoints.Count;
        }

        public override void InitializeAgent()
        {
            agentParameters = new AgentParameters();
            agentParameters.resetOnDone = true;
            agentParameters.numberOfActionsBetweenDecisions = _frameSkip;
            agentParameters.maxStep = _nSteps * _frameSkip;
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
            if (mujocoSpawner != null)
                mujocoSpawner.MujocoXml = MujocoXml;
            mujocoSpawner.SpawnFromXml();
            SetupMujoco();
            _mujocoController.UpdateFromExternalComponent();
            switch(ActorId)
            {
                case "a_oai_walker2d-v0":
                    _stepReward = StepReward_OaiWalker;
                    _terminate = Terminate_OnNonFootHitTerrain;
                    _observations = Observations_Default;
                    break;
                case "a_dm_walker-v0":
                    _stepReward = StepReward_DmWalker;
                    _terminate = Terminate_OnNonFootHitTerrain;
                    _observations = Observations_Default;
                    break;
                case "a_oai_hopper-v0":
                    _stepReward = StepReward_OaiHopper;
                    _terminate = Terminate_HopperOai;
                    _observations = Observations_Default;
                    break;
                case "a_oai_humanoid-v0":
                    _stepReward = StepReward_OaiHumanoidRun;
                    //_stepReward = StepReward_OaiHumanoidStand;
                    _terminate = Terminate_OnNonFootHitTerrain;
                    _observations = Observations_Humanoid;
                    _bodyParts["pelvis"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="butt");
                    _bodyParts["shoulders"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="torso1");
                    break;
                case "a_ant-v0":
                case "a_oai_half_cheetah-v0":
                default:
                    throw new NotImplementedException();
            }
        }
			// AddRewardFunc("a_oai_half_cheetah-v0", StepReward_VelocityIfNewMaxPosX, TerminalReward_NegNonFoot);
			// AddRewardFunc("a_oai_hopper-v0", StepReward_OaiHopper, 		TerminalReward_None);
			// AddRewardFunc("a_oai_humanoid-v0", StepReward_VelocityIfNewMaxPosX, 	TerminalReward_NegNonFoot);
			// AddRewardFunc("a_oai_humanoid-v01", StepReward_VelocityNegNonFoot, 	TerminalReward_None);
			// AddRewardFunc("a_oai_humanoid-v02", StepReward_StandNegNonFoot, 	TerminalReward_None);
			// AddRewardFunc("a_oai_walker2d-v0", StepReward_OaiWalker, 	TerminalReward_NegNonFoot);
			// AddRewardFunc("a_dm_walker-v0", StepReward_DmWalker, 	TerminalReward_None);
			// AddRewardFunc("a_RagdollSnake-v0", StepReward_VelocityIfNewMaxPosX, 	TerminalReward_None);
			// AddRewardFunc("a_RagdollSnake-v01", StepReward_NewMaxPosX, 	TerminalReward_None);
			// AddRewardFunc("a_ant-v0", StepReward_VelocityIfNewMaxPosX, 	TerminalReward_None);
			// AddRewardFunc("a_ant-v01", StepReward_NewMaxPosX, 	TerminalReward_None);
            // AddRewardFunc("a_oai_half_cheetah-v0",  Terminate_OnFall);
			// AddRewardFunc("a_oai_hopper-v0",        Terminate_HopperOai);//Terminate_OnNonFootHitTerrain);
			// AddRewardFunc("a_oai_humanoid-v0",      Terminate_OnNonFootHitTerrain);
			// AddRewardFunc("a_oai_humanoid-v01",     Terminate_OnTenErrors);
			// AddRewardFunc("a_oai_humanoid-v02",     Terminate_OnNonFootHitTerrain);
			// AddRewardFunc("a_oai_walker2d-v0",      Terminate_OnNonFootHitTerrain);
			// AddRewardFunc("a_dm_walker-v0",      	Terminate_OnNonFootHitTerrain);
			// AddRewardFunc("a_RagdollSnake-v0",      Terminate_Never);
			// AddRewardFunc("a_RagdollSnake-v01",     Terminate_Never);
			// AddRewardFunc("a_ant-v0",               Terminate_Never);
			// AddRewardFunc("a_ant-v01",              Terminate_Never);
        public override void CollectObservations()
        {
            _mujocoController.UpdateQFromExternalComponent();
            _observations();
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
            AddVectorObs(_mujocoController.JointAngles);
            AddVectorObs(_mujocoController.JointVelocity);
        }
        void Observations_Default()
        {        
            var joints = _mujocoController.MujocoJoints;

            if (ShowMonitor) {
                //Monitor.Log("pos", _mujocoController.qpos, MonitorType.hist);
                //Monitor.Log("vel", _mujocoController.qvel, MonitorType.hist);
                //Monitor.Log("onSensor", _mujocoController.OnSensor, MonitorType.hist);
                //Monitor.Log("sensor", _mujocoController.SensorIsInTouch, MonitorType.hist);
            }
           for (int j = 0; j < _numJoints; j++)
            {
                var offset = j * _jointSize;
                _observation1D[offset+00] = _mujocoController.qpos[j];
                _observation1D[offset+01] = _mujocoController.qvel[j];
                // _observation1D[offset+02] = _mujocoController.qglobpos[j];
            }
            for (int j = 0; j < _numSensors; j++)
            {
                var offset = _sensorOffset + (j * _sensorSize);
                // _observation1D[offset+00] = _mujocoController.OnSensor[j];
                _observation1D[offset+00] = _mujocoController.SensorIsInTouch[j]; // try this when using nstack
                // _observation1D[offset+01] = _mujocoController.MujocoSensors[j].SiteObject.transform.position.x;
                // _observation1D[offset+02] = _mujocoController.MujocoSensors[j].SiteObject.transform.position.y;
                // _observation1D[offset+03] = _mujocoController.MujocoSensors[j].SiteObject.transform.position.z;
            }
            _observation1D = _observation1D
                .Select(x=> UnityEngine.Mathf.Clamp(x,-5, 5))
                // .Select(x=> x / 5f)
                .ToArray();
            AddVectorObs(_observation1D);
        }

        public override void AgentAction(float[] vectorAction, string textAction)
        {
            _actions = vectorAction
                .Select(x=>Mathf.Clamp(x, -1, 1f))
                .ToList();
            //KillJointPower(new []{"shoulder", "elbow"}); // HACK
            // if (ShowMonitor)
            //     Monitor.Log("actions", _actions, MonitorType.hist);
            for (int i = 0; i < _mujocoController.MujocoJoints.Count; i++) {
				var inp = (float)_actions[i];
				MujocoController.ApplyAction(_mujocoController.MujocoJoints[i], inp);
			}
            _mujocoController.UpdateFromExternalComponent();
            
            var done = _terminate();//Terminate_HopperOai();

            if (done)
            {
                Done();
                var reward = 0;
                // var reward = -1000f;
                SetReward(reward);
            }
            if (!IsDone())
            {
                var reward = _stepReward();//StepReward_OaiHopper();
                SetReward(reward);
            }
            _footHitTerrain = false;
            _nonFootHitTerrain = false;
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
            var qpos2 = (GetAngleFromUp() % 180 ) / 180;
			var height = _mujocoController.FocalPointPosition.y - lowestFoot;
            return height;
        }
        float GetVelocity()
        {
			var dt = Time.fixedDeltaTime;
			var rawVelocity = _mujocoController.FocalPointPositionVelocity.x;
            var maxSpeed = 4f; // meters per second
            rawVelocity = Mathf.Clamp(rawVelocity,-maxSpeed,maxSpeed);
			var velocity = rawVelocity / maxSpeed;
            if (ShowMonitor) {
                Monitor.Log("rawVelocity", rawVelocity, MonitorType.text);
                Monitor.Log("velocity", velocity, MonitorType.text);
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
            var angleFromUp = Vector3.Angle(_bodyParts[bodyPart].transform.forward, Vector3.up);
            var qpos2 = (angleFromUp % 180 ) / 180;
            var uprightBonus = 0.5f * (2 - (Mathf.Abs(qpos2)*2)-1);
            // if (ShowMonitor)
                // Monitor.Log($"upright{bodyPart}Bonus", uprightBonus, MonitorType.text);
            return uprightBonus;
        }
        float GetHeightPenality(float maxHeight)
        {
            var height = GetHeight();
            var heightPenality = maxHeight - height;
			heightPenality = Mathf.Clamp(heightPenality, 0f, maxHeight);
            if (ShowMonitor) {
                Monitor.Log("height", height, MonitorType.text);
                Monitor.Log("heightPenality", heightPenality, MonitorType.text);
            }
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
                _actions[_mujocoController.MujocoJoints.IndexOf(joint)] = 0f;
        }
        float GetHumanoidArmEffort()
        {
            var mJoints = _mujocoController.MujocoJoints
                .Where(x=>x.JointName.ToLowerInvariant().Contains("shoulder") || x.JointName.ToLowerInvariant().Contains("elbow"))
                .ToList();
            var effort = mJoints
                .Select(x=>_actions[_mujocoController.MujocoJoints.IndexOf(x)])
				.Select(x=>Mathf.Pow(Mathf.Abs(x),2))
				.Sum();
            return effort;            
        }
        float GetEffort()
        {
			var effort = _actions
				.Select(x=>Mathf.Pow(Mathf.Abs(x),2))
				.Sum();
            // if (ShowMonitor)
                // Monitor.Log("effort", effort, MonitorType.text);
            return effort;
        }
        float GetEffortSum()
        {
			var effort = _actions
				.Select(x=>Mathf.Abs(x))
				.Sum();
            return effort;
        }
        float GetEffortMean()
        {
			var effort = _actions
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
        
        float StepReward_OaiHumanoidStand()
        {
            float heightPenality = GetHeightPenality(1.2f);
            float shouldersUprightBonus = GetUprightBonus("shoulders") / 2;
            float pelvisUprightBonus = GetUprightBonus("pelvis") / 2;
            float effortSquare = GetEffort();
            var effortSquarePenality = 0.02f * (float)effortSquare;
            float effortSum = GetEffortSum();
            float effortSumPenality = 0.05f * (float)effortSum;
            effortSumPenality =  Mathf.Clamp(effortSumPenality, 0f, 2f);
			var reward = 0 
                + shouldersUprightBonus
                + pelvisUprightBonus
                - heightPenality
                - effortSquarePenality
			    - effortSumPenality;
            if (ShowMonitor) {
                var hist = new []{reward, shouldersUprightBonus, pelvisUprightBonus,- heightPenality,-effortSquarePenality,-effortSumPenality}.ToList();
                Monitor.Log("rewardHist", hist, MonitorType.hist);
                //Monitor.Log("effortPenality", effortPenality, MonitorType.text);
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
        
        float StepReward_OaiHumanoidRun()
        {
            float velocity = GetVelocity();
            float heightPenality = GetHeightPenality(1.2f);
            float shouldersUprightBonus = GetUprightBonus("shoulders") / 2;
            float pelvisUprightBonus = GetUprightBonus("pelvis") / 2;
            float effort = GetEffort();
            var effortPenality = 0.02f * (float)effort;
            var armPenalty = 0.1f * (float)GetHumanoidArmEffort();
			var reward = velocity 
                + shouldersUprightBonus
                + pelvisUprightBonus
                - heightPenality
			    - effortPenality
                - armPenalty;
            if (ShowMonitor) {
                var hist = new []{reward,velocity, shouldersUprightBonus, pelvisUprightBonus,- heightPenality,-effortPenality, -armPenalty}.ToList();
                Monitor.Log("rewardHist", hist, MonitorType.hist);
                //Monitor.Log("effortPenality", effortPenality, MonitorType.text);
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
			return _nonFootHitTerrain;
		}
        bool Terminate_HopperOai()
		{
			if (_nonFootHitTerrain)
				return true;
			if (_mujocoController.qpos == null)
				return false;
			var height = _mujocoController.qpos[1];
			var angle = Mathf.Abs(_mujocoController.qpos[2]);
			bool endOnHeight = (height < .3f);
			bool endOnAngle = (angle > (1f/180f) * (5.7296f *6));
			return endOnHeight || endOnAngle;
		}

		public void OnTerrainCollision(GameObject other, GameObject terrain) {
            if (string.Compare(terrain.name, "Terrain", true) != 0)
                return;
            
            switch (other.name.ToLowerInvariant().Trim())
            {
                case "left_foot": // oai_humanoid
                case "right_foot": // oai_humanoid
                case "right_shin1": // oai_humanoid
                case "left_shin1": // oai_humanoid
                case "foot_geom": // oai_hopper  //oai_walker2d
                case "leg_geom": // oai_hopper //oai_walker2d
                case "leg_left_geom": // oai_walker2d
                case "foot_left_geom": //oai_walker2d
                case "foot_left_joint": //oai_walker2d
                case "foot_joint": //oai_walker2d
                    _footHitTerrain = true;
                    break;
                default:
                    _nonFootHitTerrain = true;
                    break;
            }
		}            
    }
}