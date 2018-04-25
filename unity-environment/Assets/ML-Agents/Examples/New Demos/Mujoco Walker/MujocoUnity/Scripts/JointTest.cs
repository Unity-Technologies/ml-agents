using UnityEngine;

namespace MujocoUnity {

    static class LocalExt {
        public static void MoveRotationTorque (this Rigidbody rigidbody, Quaternion targetRotation) {
            rigidbody.maxAngularVelocity = 1000;

            Quaternion rotation = targetRotation * Quaternion.Inverse (rigidbody.rotation);
            rigidbody.AddTorque (rotation.x / Time.fixedDeltaTime, rotation.y / Time.fixedDeltaTime, rotation.z / Time.fixedDeltaTime, ForceMode.VelocityChange);
            rigidbody.angularVelocity = Vector3.zero;
        }

    }

    public class JointTest : MonoBehaviour {
        // public HingeJoint hj;
        public HingeJoint joint1;
        public HingeJoint joint2;
        public HingeJoint joint3;

        public bool applyRandomToAll;
        // public Vector3 force;
        public float target1;
        public float target2;
        public float target3;

        void Start () {

        }

        void ApplyTarget (HingeJoint joint, float target) {
            if (joint != null) {
                JointSpring js;
                js = joint.spring;
                var safeTarget = Mathf.Clamp (target, 0, 1);
                var min = joint.limits.min;
                var max = joint.limits.max;
                var scale = max - min;
                var scaledTarget = min + (safeTarget * scale);
                js.targetPosition = scaledTarget;
                joint.spring = js;
            }
        }

        void FixedUpdate () {
            if (applyRandomToAll) {
                ApplyTarget (joint1, Random.value);
                ApplyTarget (joint2, Random.value);
                ApplyTarget (joint3, Random.value);
            } else {
                ApplyTarget (joint1, target1);
                ApplyTarget (joint2, target2);
                ApplyTarget (joint3, target3);
            }
        }
    }
}