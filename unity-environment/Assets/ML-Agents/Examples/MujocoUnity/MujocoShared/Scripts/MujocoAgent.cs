using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace MujocoUnity
{
    public class MujocoAgent : Agent
    {
        public bool FootHitTerrain;
        public bool NonFootHitTerrain;
        public float VelocityScaler = 16f; //50f;//16f;//300;//1500f; 
        

        public float[] Low;
		public float[] High;
        public bool ShowMonitor;
		float[] _observation1D;
        float[] _internalLow;
        float[] _internalHigh;
        int _jointSize = 13; // 9+4
        int _numJoints = 3; // for debug object
        int _sensorOffset; // offset in observations to where senors begin
        internal int NumSensors;
        int _sensorSize; // number of floats per senor
        int _observationSize; // total number of floats
        

        protected MujocoController MujocoController;

        public List<float> Actions;
        protected Func<bool> TerminateFunction;
        protected Func<float> StepRewardFunction;
        protected Action ObservationsFunction;
        protected Dictionary<string,Rigidbody> BodyParts = new Dictionary<string,Rigidbody>();
        protected Dictionary<string,Quaternion> BodyPartsToFocalRoation = new Dictionary<string,Quaternion>();    

    Dictionary<GameObject, Vector3> transformsPosition;
    Dictionary<GameObject, Quaternion> transformsRotation;
    
    MujocoSpawner mujocoSpawner;    

        public override void AgentReset()
        {
            if (mujocoSpawner == null)
                mujocoSpawner = this.GetComponent<MujocoUnity.MujocoSpawner>();

            Transform[] allChildren = GetComponentsInChildren<Transform>();
            if (MujocoController != null) {
                // restore
                foreach (Transform child in allChildren)
                {
                    if (child.gameObject.name.Contains("OpenAIHumanoid"))
                    {
                        continue;
                    }
                    child.position = transformsPosition[child.gameObject];
                    child.rotation = transformsRotation[child.gameObject];
                    var childRb = child.GetComponent<Rigidbody>();
                    if (childRb != null) {
                        childRb.angularVelocity = Vector3.zero;
                        childRb.velocity = Vector3.zero;
                    }
                }
                mujocoSpawner.ApplyRandom();
                SetupMujoco();
                MujocoController.UpdateFromExternalComponent();
                return;
            }
            MujocoController = GetComponent<MujocoController>();
            MujocoController.MujocoJoints = null;
            MujocoController.MujocoSensors = null;
            // var joints = this.GetComponentsInChildren<Joint>().ToList();
            // foreach (var item in joints)
            //     Destroy(item.gameObject);
            var rbs = this.GetComponentsInChildren<Rigidbody>().ToList();
            foreach (var item in rbs){
                if (item != null) 
                    DestroyImmediate(item.gameObject);
            }
            Resources.UnloadUnusedAssets();

            mujocoSpawner.SpawnFromXml();
            allChildren = GetComponentsInChildren<Transform>();
            transformsPosition = new Dictionary<GameObject, Vector3>();
            transformsRotation = new Dictionary<GameObject, Quaternion>();
            foreach (Transform child in allChildren)
            {
                transformsPosition[child.gameObject] = child.position;
                transformsRotation[child.gameObject] = child.rotation;
            }
            mujocoSpawner.ApplyRandom();
            SetupMujoco();
            MujocoController.UpdateFromExternalComponent();
        }
        void SetupMujoco()
        {
            MujocoController = GetComponent<MujocoController>();
            _numJoints = MujocoController.MujocoJoints.Count;
            NumSensors = MujocoController.MujocoSensors.Count;            
            _jointSize = 2;
            _sensorSize = 1;
            _sensorOffset = _jointSize * _numJoints;
            _observationSize = _sensorOffset + (_sensorSize * NumSensors);
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
            for (int j = 0; j < NumSensors; j++)
            {
                var offset = _sensorOffset + (j * _sensorSize);
                _internalLow[offset+0] = -1;//-10;
                _internalHigh[offset+0] = 1;//10;
            }    
            //this.brain = GetComponent<Brain>();
        }        
        internal void SetupBodyParts()
        {
            // set body part directions
            foreach (var bodyPart in BodyParts)
            {
                var name = bodyPart.Key;
                var rigidbody = bodyPart.Value;

                // find up
                var focalPoint = rigidbody.position;
                focalPoint.x += 10;
                var focalPointRotation = rigidbody.rotation;
                focalPointRotation.SetLookRotation(focalPoint - rigidbody.position);
                BodyPartsToFocalRoation[name] = focalPointRotation;
            }
        }

        public override void CollectObservations()
        {
            MujocoController.UpdateQFromExternalComponent();
            ObservationsFunction();
        }
        public override void AgentAction(float[] vectorAction, string textAction)
        {
            Actions = vectorAction
                .Select(x=>x)
                .ToList();
            for (int i = 0; i < MujocoController.MujocoJoints.Count; i++) {
                var inp = (float)Actions[i];
                ApplyAction(MujocoController.MujocoJoints[i], inp);
            }
            MujocoController.UpdateFromExternalComponent();
            
            var done = TerminateFunction();

            if (done)
            {
                Done();
                var reward = -1f;
                SetReward(reward);
            }
            if (!IsDone())
            {
                var reward = StepRewardFunction();
                SetReward(reward);
            }

            FootHitTerrain = false;
            NonFootHitTerrain = false;
        }

        internal void KillJointPower(string[] hints)
        {
            var mJoints = hints
                .SelectMany(hint=>
                    MujocoController.MujocoJoints
                        .Where(x=>x.JointName.ToLowerInvariant().Contains(hint.ToLowerInvariant()))
                ).ToList();
            foreach (var joint in mJoints)
                Actions[MujocoController.MujocoJoints.IndexOf(joint)] = 0f;
        }

        internal float GetHeight()
        {
			var feetYpos = MujocoController.MujocoJoints
				.Where(x=>x.JointName.ToLowerInvariant().Contains("foot"))
				.Select(x=>x.Joint.transform.position.y)
				.OrderBy(x=>x)
				.ToList();
            float lowestFoot = 0f;
            if(feetYpos!=null && feetYpos.Count != 0)
                lowestFoot = feetYpos[0];
			var height = MujocoController.FocalPoint.transform.position.y - lowestFoot;
            return height;
        }
        internal float GetVelocity()
        {
			var dt = Time.fixedDeltaTime;
			var rawVelocity = MujocoController.FocalRidgedBody.velocity.x;
            var maxSpeed = 4f; // meters per second
            //rawVelocity = Mathf.Clamp(rawVelocity,-maxSpeed,maxSpeed);
			var velocity = rawVelocity / maxSpeed;
            if (ShowMonitor) {
                Monitor.Log("MaxDistance", MujocoController.FocalPointMaxDistanceTraveled);
                Monitor.Log("MPH: ", rawVelocity * 2.236936f, MonitorType.text);
                // Monitor.Log("rawVelocity", rawVelocity, MonitorType.text);
                // Monitor.Log("velocity", velocity, MonitorType.text);
            }
            return velocity;
        }
        internal float GetUprightBonus()
        {
            var qpos2 = (GetAngleFromUp() % 180 ) / 180;
            var uprightBonus = 0.5f * (2 - (Mathf.Abs(qpos2)*2)-1);
            // if (ShowMonitor)
                // Monitor.Log("uprightBonus", uprightBonus, MonitorType.text);
            return uprightBonus;
        }
        internal float GetUprightBonus(string bodyPart)
        {
            var toFocalAngle = BodyPartsToFocalRoation[bodyPart] * -BodyParts[bodyPart].transform.forward;
            var angleFromUp = Vector3.Angle(toFocalAngle, Vector3.up);
            var qpos2 = (angleFromUp % 180 ) / 180;
            var uprightBonus = 0.5f * (2 - (Mathf.Abs(qpos2)*2)-1);
            // if (ShowMonitor)
            //     Monitor.Log($"upright[{bodyPart}] Bonus", uprightBonus, MonitorType.text);
            return uprightBonus;
        }

        internal float GetDirectionBonus(string bodyPart, Vector3 direction, float maxBonus = 0.5f)
        {
            var toFocalAngle = BodyPartsToFocalRoation[bodyPart] * BodyParts[bodyPart].transform.right;
            var angle = Vector3.Angle(toFocalAngle, direction);
            var qpos2 = (angle % 180 ) / 180;
            var bonus = maxBonus * (2 - (Mathf.Abs(qpos2)*2)-1);
            return bonus;
        }
        internal void GetDirectionDebug(string bodyPart)
        {
            var toFocalAngle = BodyPartsToFocalRoation[bodyPart] * BodyParts[bodyPart].transform.right;
            var angleFromLeft = Vector3.Angle(toFocalAngle, Vector3.left);
            var angleFromUp = Vector3.Angle(toFocalAngle, Vector3.up);
            var angleFromDown = Vector3.Angle(toFocalAngle, Vector3.down);
            var angleFromRight = Vector3.Angle(toFocalAngle, Vector3.right);
            var angleFromForward = Vector3.Angle(toFocalAngle, Vector3.forward);
            var angleFromBack = Vector3.Angle(toFocalAngle, Vector3.back);
            print ($"{bodyPart}: l: {angleFromLeft}, r: {angleFromRight}, f: {angleFromForward}, b: {angleFromBack}, u: {angleFromUp}, d: {angleFromDown}");
        }

        internal float GetLeftBonus(string bodyPart)
        {
            var bonus = GetDirectionBonus(bodyPart, Vector3.left);
            // if (ShowMonitor)
            //     Monitor.Log($"left[{bodyPart}] Bonus", bonus, MonitorType.text);
            // print (bonus);
            return bonus;
        }       
        internal float GetRightBonus(string bodyPart)
        {
            var bonus = GetDirectionBonus(bodyPart, Vector3.right);
            // if (ShowMonitor)
            //     Monitor.Log($"right[{bodyPart}] Bonus", bonus, MonitorType.text);
            // print (bonus);
            return bonus;
        }       
        internal float GetForwardBonus(string bodyPart)
        {
            var bonus = GetDirectionBonus(bodyPart, Vector3.forward);
            // if (ShowMonitor)
            //     Monitor.Log($"forward[{bodyPart}] Bonus", bonus, MonitorType.text);
            // print (bonus);
            return bonus;
        }
        internal float GetHeightPenality(float maxHeight)
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
        internal float GetEffort(string[] ignorJoints = null)
        {
            double effort = 0;
            for (int i = 0; i < Actions.Count; i++)
            {
                var name = MujocoController.MujocoJoints[i].JointName;
                if (ignorJoints != null && ignorJoints.Contains(name))
                    continue;
                var jointEffort = Mathf.Pow(Mathf.Abs(Actions[i]),2);
                effort += jointEffort;
            }
            return (float)effort;
        }
        internal float GetJointsAtLimitPenality(string[] ignorJoints = null)
        {
            int atLimitCount = 0;
            for (int i = 0; i < Actions.Count; i++)
            {
                var name = MujocoController.MujocoJoints[i].JointName;
                if (ignorJoints != null && ignorJoints.Contains(name))
                    continue;
                bool atLimit = Mathf.Abs(Actions[i]) >= 1f;
                if (atLimit)
                    atLimitCount++;
            }
            float penality = atLimitCount * 0.2f;
            return (float)penality;            
        }
        internal float GetEffortSum()
        {
			var effort = Actions
				.Select(x=>Mathf.Abs(x))
				.Sum();
            return effort;
        }
        internal float GetEffortMean()
        {
			var effort = Actions
				.Average();
            return effort;
        }

        internal float GetAngleFromUp()
        {
            var angleFromUp = Vector3.Angle(MujocoController.FocalPoint.transform.forward, Vector3.up);
            if (ShowMonitor) {
                // Monitor.Log("AngleFromUp", angleFromUp);
            }
            return angleFromUp; 
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
                case "right_leg": // dm_walker
                case "left_leg": // dm_walker
                    FootHitTerrain = true;
                    break;
                default:
                    NonFootHitTerrain = true;
                    break;
            }
        }         
        internal bool TerminateNever()
        {
            return false;
        }
        internal bool TerminateOnNonFootHitTerrain()
        {
            return NonFootHitTerrain;
        }  

		internal void ApplyAction(MujocoJoint mJoint, float? target = null)
        {
            ConfigurableJoint configurableJoint = mJoint.Joint as ConfigurableJoint;
            if (!target.HasValue) // handle random
                target = UnityEngine.Random.value * 2 - 1;
            var t = configurableJoint.targetAngularVelocity;
            t.x = target.Value * VelocityScaler;
            configurableJoint.targetAngularVelocity = t;
            var angX = configurableJoint.angularXDrive;
            angX.positionSpring = 1f;
            var scale = mJoint.MaximumForce * Mathf.Pow(Mathf.Abs(target.Value),3);
            angX.positionDamper = Mathf.Max(1f, scale);
            angX.maximumForce = Mathf.Max(1f, scale);
            configurableJoint.angularXDrive = angX;
        }          
    }
}