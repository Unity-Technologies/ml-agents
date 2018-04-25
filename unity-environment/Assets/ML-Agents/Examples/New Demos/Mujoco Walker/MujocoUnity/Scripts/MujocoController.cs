using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace MujocoUnity 
{
    public class MujocoController : MonoBehaviour
    {
        public List<MujocoJoint> MujocoJoints;
        public List<MujocoSensor> MujocoSensors;
        
        public GameObject CameraTarget;

        public bool applyRandomToAll;
        public bool applyTargets;
        public float[] targets;

        public List<float> qpos;
        public List<float> qglobpos;
        public List<float> qvel;
        static float _velocityScaler = 50f; //50f;//16f;//300;//1500f; 
        public List<float> OnSensor;
        public List<float> SensorIsInTouch;
        public float MujocoTimeStep;

        public GameObject _focalPoint;

        public Vector3 FocalPointPosition;
        public Vector3 FocalPointPositionVelocity;
        public Vector3 FocalPointRotation;
        public Vector3 FocalPointEulerAngles;
        public Vector3 FocalPointRotationVelocity;
        public List<float> JointAngles;
        public List<float> JointVelocity;
        


        bool _externalMode;

        public void SetMujocoSensors(List<MujocoSensor> mujocoSensors)
        {
            MujocoSensors = mujocoSensors;
            OnSensor = Enumerable.Range(0,mujocoSensors.Count).Select(x=>0f).ToList();
            SensorIsInTouch = Enumerable.Range(0,mujocoSensors.Count).Select(x=>0f).ToList();
            foreach (var sensor in mujocoSensors)
            {
                sensor.SiteObject.gameObject.AddComponent<SensorBehavior>();
            }
        }

        public void SetMujocoJoints(List<MujocoJoint> mujocoJoints)
        {
            MujocoJoints = mujocoJoints;
            targets = Enumerable.Repeat(0f, MujocoJoints.Count).ToArray();
            var target = FindTopMesh(MujocoJoints.FirstOrDefault()?.Joint.gameObject, null);
            if (CameraTarget != null && MujocoJoints != null) {
                var smoothFollow = CameraTarget.GetComponent<SmoothFollow>();
                if (smoothFollow != null) 
                    smoothFollow.target = target.transform;
            }
            _focalPoint = target;
            var qlen = MujocoJoints.Count + 3;
            qpos = Enumerable.Range(0,qlen).Select(x=>0f).ToList();
            qglobpos = Enumerable.Range(0,qlen).Select(x=>0f).ToList();
            qvel = Enumerable.Range(0,qlen).Select(x=>0f).ToList();
            JointAngles = Enumerable.Range(0,MujocoJoints.Count).Select(x=>0f).ToList();
            JointVelocity = Enumerable.Range(0,MujocoJoints.Count).Select(x=>0f).ToList();
        }

        public void SetMujocoTimestep(float timestep)
        {
            MujocoTimeStep = timestep;
        }
        GameObject FindTopMesh(GameObject curNode, GameObject topmostNode = null)
        {
            var meshRenderer = curNode.GetComponent<MeshRenderer>();
            if (meshRenderer != null)
                topmostNode = meshRenderer.gameObject;
            var root = curNode.transform.root.gameObject;
            var meshRenderers = root.GetComponentsInChildren<MeshRenderer>();
            if (meshRenderers != null && meshRenderers.Length >0)
                topmostNode = meshRenderers[0].gameObject;
            
            // var parent = curNode.transform.parent;//curNode.GetComponentInParent<Transform>()?.gameObject;
            // if (parent != null)
            //     return FindTopMesh(curNode, topmostNode);
            return (topmostNode);
            
        }
        void Start () {
            
        }
        
        // Update is called once per frame
        void Update () 
        {
            if(_externalMode)
                return;
            if (MujocoJoints == null || MujocoJoints.Count ==0)
                return;
            for (int i = 0; i < MujocoJoints.Count; i++)
            {
                if (applyRandomToAll)
                    ApplyAction(MujocoJoints[i]);
                else if (applyTargets)
                    ApplyAction(MujocoJoints[i], targets[i]);
            }
            UpdateQ(true);
        }

        void LateUpdate()
        {
            if (_externalMode)
                return;
            for (int i = 0; i < OnSensor.Count; i++)
                OnSensor[i] = 0f;
        }
        public void UpdateFromExternalComponent(bool useDeltaTime = false)
        {
            for (int i = 0; i < OnSensor.Count; i++)
                OnSensor[i] = 0f;
            _externalMode = true;
            if (MujocoJoints == null || MujocoJoints.Count ==0)
                return;
            for (int i = 0; i < MujocoJoints.Count; i++)
            {
                if (applyRandomToAll)
                    ApplyAction(MujocoJoints[i]);
                else if (applyTargets)
                    ApplyAction(MujocoJoints[i], targets[i]);
            }
            UpdateQ(useDeltaTime);
        }
        public void UpdateQFromExternalComponent(bool useDeltaTime = false)
        {
            UpdateQ(useDeltaTime);
        }
        void UpdateQ(bool useDeltaTime = false)
        {
			float dt = Time.fixedDeltaTime;
            if(useDeltaTime)
                dt = Time.deltaTime;

            var focalTransform = _focalPoint.transform;
            var focalRidgedBody = _focalPoint.GetComponent<Rigidbody>();
            FocalPointPosition = focalTransform.position;
            FocalPointPositionVelocity = focalRidgedBody.velocity;
            var lastFocalPointRotationVelocity = FocalPointRotation;
            FocalPointEulerAngles = focalTransform.eulerAngles;
            FocalPointRotation = new Vector3(
                ((FocalPointEulerAngles.x - 180f) % 180 ) / 180,
                ((FocalPointEulerAngles.y - 180f) % 180 ) / 180,
                ((FocalPointEulerAngles.z - 180f) % 180 ) / 180);
            FocalPointRotationVelocity = (FocalPointRotation-lastFocalPointRotationVelocity);
                        
            var topJoint = MujocoJoints[0];
            //var topTransform = topJoint.Joint.transform.parent.transform;
            // var topRidgedBody = topJoint.Joint.transform.parent.GetComponent<Rigidbody>();
            var topTransform = topJoint.Joint.transform;
            var topRidgedBody = topJoint.Joint.transform.GetComponent<Rigidbody>();
            qpos[0] = topTransform.position.x;
            qglobpos[0] = topTransform.position.x;     
            qvel[0] = topRidgedBody.velocity.x;
            qpos[1] = topTransform.position.y;
            qglobpos[1] = topTransform.position.y;
            qvel[1] = topRidgedBody.velocity.y;
            qpos[2] = ((topTransform.rotation.eulerAngles.z - 180f) % 180 ) / 180;
            qglobpos[2] = ((topTransform.rotation.eulerAngles.z - 180f) % 180 ) / 180;
            qvel[2] = topRidgedBody.velocity.z;
            for (int i = 0; i < MujocoJoints.Count; i++)
            {
                var joint = MujocoJoints[i].Joint;
                // var targ = joint.transform.parent.transform;
                var targ = joint.transform;
                float pos = 0f;
                float globPos = 0f;
                if (joint.axis.x != 0f) {
                    pos = targ.localEulerAngles.x;
                    globPos = targ.eulerAngles.x;
                }
                else if (joint.axis.y != 0f){
                    pos = targ.localEulerAngles.y;
                    globPos = targ.eulerAngles.y;
                }
                else if (joint.axis.z != 0f) {
                    pos = targ.localEulerAngles.z;
                    globPos = targ.eulerAngles.z;
                }
                HingeJoint hingeJoint = joint as HingeJoint;
                ConfigurableJoint configurableJoint = joint as ConfigurableJoint;
                pos = ((pos - 180f) % 180 ) / 180;
                // pos /= 180f;
                globPos = ((globPos - 180f) % 180 ) / 180;
                if (hingeJoint != null){
                    qpos[3+i] = pos;
                    qglobpos[3+i] = globPos;
                    qvel[3+i] = hingeJoint.velocity / _velocityScaler;
                }
                else if (configurableJoint != null){
                    var lastPos = qpos[3+i];
                    qpos[3+i] = pos;
                    JointAngles[i] = pos;
                    var lastgPos = qglobpos[3+i];
                    qglobpos[3+i] = globPos;
                    // var vel = joint.gameObject.GetComponent<Rigidbody>().velocity.x;
                    // var vel = configurableJoint.
                    var vel = (qpos[3+i] - lastPos) / (dt);
                    qvel[3+i] = vel;
                    JointVelocity[i] = vel;
                }

            }
            
        }

		static public void ApplyAction(MujocoJoint mJoint, float? target = null)
        {
            HingeJoint hingeJoint = mJoint.Joint as HingeJoint;
            ConfigurableJoint configurableJoint = mJoint.Joint as ConfigurableJoint;
            if (configurableJoint != null){
                if (!target.HasValue) // handle random
                    target = Random.value * 2 - 1;
                target = Mathf.Clamp(target.Value, -1f, 1f);
                var t = configurableJoint.targetAngularVelocity;
                t.x = target.Value * _velocityScaler;
                configurableJoint.targetAngularVelocity = t;
                return;
            } else if (hingeJoint == null)
                return;
            if (hingeJoint.useSpring)
            {
                var ctrlRangeMin = -1f;
                var ctrlRangeMax = 1f;
                // var ctrlRangeMin = 0f;
                // var ctrlRangeMax = 1f;
                var inputScale = ctrlRangeMax - ctrlRangeMin;
                if (!target.HasValue) // handle random
                    target = ctrlRangeMin + (Random.value * inputScale);
                var inputTarget = Mathf.Clamp(target.Value, ctrlRangeMin, ctrlRangeMax);
                if (ctrlRangeMin < 0)
                    inputTarget = Mathf.Abs(ctrlRangeMin) + inputTarget;
                else
                    inputTarget = inputTarget - Mathf.Abs(ctrlRangeMin);
                inputTarget /= inputScale;
                JointSpring js;
                js = hingeJoint.spring;
                var min = hingeJoint.limits.min;
                var max = hingeJoint.limits.max;
                var outputScale = max-min;
                var outputTarget = min+(inputTarget * outputScale);
                js.targetPosition = outputTarget;
                hingeJoint.spring = js;
            }
            else if (hingeJoint.useMotor)
            {
                if (!target.HasValue) // handle random
                    target = Random.value * 2 - 1;

                target = Mathf.Clamp(target.Value, -1f, 1f);
                // target = Mathf.Clamp(target.Value, 0f, 1f);
                // target *= 2;
                // target -= 1f;

                JointMotor jm;
                jm = hingeJoint.motor;
                jm.targetVelocity = target.Value * _velocityScaler;
                hingeJoint.motor = jm;
            }
        }

        public void SensorCollisionEnter(Collider sensorCollider, Collision other) {
			if (string.Compare(other.gameObject.name, "Terrain", true) !=0)
                return;
			var otherGameobject = other.gameObject;
            var sensor = MujocoSensors
                .FirstOrDefault(x=>x.SiteObject == sensorCollider);
            if (sensor != null) {
                var idx = MujocoSensors.IndexOf(sensor);
                OnSensor[idx] = 1f;
                SensorIsInTouch[idx] = 1f;
            }
		}
        public void SensorCollisionExit(Collider sensorCollider, Collision other)
        {
            if (string.Compare(other.gameObject.name, "Terrain", true) !=0)
                return;
			var otherGameobject = other.gameObject;
            var sensor = MujocoSensors
                .FirstOrDefault(x=>x.SiteObject == sensorCollider);
            if (sensor != null) {
                var idx = MujocoSensors.IndexOf(sensor);
                OnSensor[idx] = 0f;
                SensorIsInTouch[idx] = 0f;
            }
        }
    }
}