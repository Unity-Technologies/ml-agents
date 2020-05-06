using UnityEngine;

namespace  MLAgentsExamples
{
    /// <summary>
    /// This class contains logic for common ragdoll operations & helper methods.
    /// </summary>
    public static class RagdollHelpers 
    {
        /// <summary>
        //Get Joint Rotation Relative to the Connected Rigidbody
        //We want to collect this info because it is the actual rotation, not the "target rotation"
        //..because when the joint is weak, the target rotation will be much different than the actual rotation
        /// </summary>
        public static Quaternion GetJointRotation(ConfigurableJoint joint)
        {
            Quaternion rotDiff = Quaternion.Inverse(joint.connectedBody.transform.rotation) * joint.transform.rotation;
            return(rotDiff);
    //        return(Quaternion.FromToRotation(joint.axis, joint.connectedBody.transform.rotation.eulerAngles));
        }

        public static Quaternion GetRotationDelta(Quaternion r1, Quaternion r2)
        {
            Quaternion rotDiff = Quaternion.Inverse(r1) * r2;
            return(rotDiff);
        }
    }
}