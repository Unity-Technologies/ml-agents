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

        // TODO move to some debug controller
        // public bool applyRandomToAll;
        // public bool applyTargets;
        // public float[] targets;

        [System.Obsolete]
        List<float> qpos;
        [System.Obsolete]
        List<float> qglobpos;
        [System.Obsolete]
        List<float> qvel;
        public List<float> SensorIsInTouch;

        public GameObject FocalPoint;
        public Rigidbody FocalRidgedBody;
        public float FocalPointMaxDistanceTraveled;
        [System.Obsolete]
        List<float> JointAngles;
        public List<float> JointVelocity;

        List<System.Tuple<ConfigurableJoint, Transform>> _baseTargetPairs;
        public List<Quaternion> JointRotations;
        List<Vector3> JointAngularVelocities;
        
        public void SetMujocoSensors(List<MujocoSensor> mujocoSensors)
        {
            MujocoSensors = mujocoSensors;
            SensorIsInTouch = Enumerable.Range(0,mujocoSensors.Count).Select(x=>0f).ToList();
            foreach (var sensor in mujocoSensors)
            {
                sensor.SiteObject.gameObject.AddComponent<SensorBehavior>();
            }
        }

        public void SetMujocoJoints(List<MujocoJoint> mujocoJoints)
        {
            MujocoJoints = mujocoJoints;
            // targets = Enumerable.Repeat(0f, MujocoJoints.Count).ToArray();
            var target = FindTopMesh(MujocoJoints.FirstOrDefault()?.Joint.gameObject, null);
            if (CameraTarget != null && MujocoJoints != null) {
                var smoothFollow = CameraTarget.GetComponent<SmoothFollow>();
                if (smoothFollow != null) 
                    smoothFollow.target = target.transform;
            }
            FocalPoint = target;
            FocalRidgedBody = FocalPoint.GetComponent<Rigidbody>();            
            var qlen = MujocoJoints.Count + 3;
            qpos = Enumerable.Range(0,qlen).Select(x=>0f).ToList();
            qglobpos = Enumerable.Range(0,qlen).Select(x=>0f).ToList();
            qvel = Enumerable.Range(0,qlen).Select(x=>0f).ToList();
            JointAngles = Enumerable.Range(0,MujocoJoints.Count).Select(x=>0f).ToList();
            JointVelocity = Enumerable.Range(0,MujocoJoints.Count).Select(x=>0f).ToList();
            _baseTargetPairs = MujocoJoints
                .Select(x=> new System.Tuple<ConfigurableJoint, Transform>(x.TrueBase, x.TrueTarget))
                .Distinct()
                .ToList();
            JointRotations = Enumerable.Range(0,_baseTargetPairs.Count).Select(x=>Quaternion.identity).ToList();
            JointAngularVelocities = Enumerable.Range(0,_baseTargetPairs.Count).Select(x=>Vector3.zero).ToList();
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
            // for (int i = 0; i < MujocoJoints.Count; i++)
            // {
            //     if (applyRandomToAll)
            //         ApplyAction(MujocoJoints[i]);
            //     else if (applyTargets)
            //         ApplyAction(MujocoJoints[i], targets[i]);
            // }
        }

        void LateUpdate()
        {

        }
        public void UpdateFromExternalComponent()
        {
            if (MujocoJoints == null || MujocoJoints.Count ==0)
                return;
            UpdateQ();
        }
        public void UpdateQFromExternalComponent()
        {
            UpdateQ();
        }
        void UpdateQ()
        {
			float dt = Time.fixedDeltaTime;
            FocalPointMaxDistanceTraveled = Mathf.Max(FocalPointMaxDistanceTraveled, FocalPoint.transform.position.x);

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
                // UpdateJointAngle(MujocoJoints[i], dt);
                // qpos[3+i] = JointAngles[i] = MujocoJoints[i].AngleWithinRange;
                // qvel[3+i] = JointVelocity[i] = MujocoJoints[i].AngularVelocityPerSecond / 1000;

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
                ConfigurableJoint configurableJoint = joint as ConfigurableJoint;
                pos = ((pos - 180f) % 180 ) / 180;
                // pos /= 180f;
                globPos = ((globPos - 180f) % 180 ) / 180;
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

            for (int i = 0; i < _baseTargetPairs.Count; i++)
            {
                var x = _baseTargetPairs[i];
                var baseRot = x.Item1.transform.rotation;
                var targetRot = x.Item2.rotation;                    
                var rotation = Quaternion.Inverse(baseRot) * targetRot;
                JointRotations[i] = rotation;

                var baseAngVel = x.Item1.GetComponent<Rigidbody>().angularVelocity;
                var targetAngVel = x.Item2.GetComponent<Rigidbody>().angularVelocity;
                var angVel = baseAngVel-targetAngVel;
                angVel /= dt;
                angVel /= 10000f;
                JointAngularVelocities[i] = angVel;
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
                SensorIsInTouch[idx] = 0f;
            }
        }
    }
}