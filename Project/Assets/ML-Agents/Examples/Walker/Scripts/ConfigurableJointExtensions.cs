using UnityEngine;
public static class ConfigurableJointExtensions
{
    public static Quaternion GetCurrentRotation(this ConfigurableJoint joint)
    {
        // TODO this may or may not be correct. Good to have in the long run. Need to fix.
        var currentRotation = joint.transform.rotation;
        var parentRotation = joint.transform.parent.rotation;
        var right = joint.axis;
        var forward = Vector3.Cross(joint.axis, joint.secondaryAxis).normalized;
        var up = Vector3.Cross(forward, right).normalized;
        Quaternion worldToJointSpace = Quaternion.LookRotation(forward, up);
        var r = Quaternion.Inverse(worldToJointSpace) * Quaternion.Inverse(currentRotation) * parentRotation * worldToJointSpace;
        return r;
    }
}
