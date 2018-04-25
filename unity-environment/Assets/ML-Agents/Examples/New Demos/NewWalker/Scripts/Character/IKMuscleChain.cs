using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class JointDriveInfo
{
    public float spring;
    public float damper;
    public float maxForce;
}

public class IKMuscleChain : MonoBehaviour
{
    protected Ragdoll guy;

    public Transform targetTransform;  // the target for the IK chain to reach

    public Transform relativeTo;

    public Transform actuatorLimb; // the physical limb at the end of the actual muscle chain
    public Transform ikHand;  // the IK's hand after solving

    public Transform upperFollowTransform, lowerFollowTransform;
    public ConfigurableJoint upperMuscle, lowerMuscle;

    

    Quaternion upperArmRotationOffset, lowerArmRotationOffset;










    Vector3 targetPosInChestLocalSpace;
    public void SetTargetPos(Vector3 target, Space space = Space.Self)
    {
        if (space == Space.Self)
        {
            targetPosInChestLocalSpace = target;// guy.chest.transform.TransformPoint(target);
            targetTransform.localPosition = target;
        }
        else
        {
            targetPosInChestLocalSpace = relativeTo.InverseTransformPoint(target);
            //targetTransform.position = target;
        }


    }


    Vector3 targetWorldPos;






    private void Awake()
    {
        guy = GetComponentInParent<Ragdoll>();
        upperArmRotationOffset = upperFollowTransform.localRotation;
        lowerArmRotationOffset = lowerFollowTransform.localRotation;

       
        targetPosInChestLocalSpace = relativeTo.InverseTransformPoint(targetTransform.position);
    }

    // Use this for initialization

    public float GetRotationDifferenceBetweenTargetAndActual()
    {
        return currentUpperRotationDifference + currentLowerRotationDifference;
    }

    float currentUpperRotationDifference, currentLowerRotationDifference;
    public bool debugDontLerpTarget;
    protected virtual void RunIkMuscleFollow()
    {

        if (upperMuscle != null)
        {
            Quaternion IkRelativeRotationFromStartUpper = Quaternion.Inverse(upperArmRotationOffset) * upperFollowTransform.localRotation;
            upperMuscle.targetRotation = Quaternion.RotateTowards(upperMuscle.targetRotation, Quaternion.Inverse(IkRelativeRotationFromStartUpper), 1080f * Time.deltaTime);

            currentUpperRotationDifference = Quaternion.Angle(upperMuscle.targetRotation, Quaternion.Inverse(IkRelativeRotationFromStartUpper));
        }
        if (lowerMuscle != null)
        {
            Quaternion IkRelativeRotationFromStartLower = Quaternion.Inverse(lowerArmRotationOffset) * lowerFollowTransform.localRotation;
            lowerMuscle.targetRotation = Quaternion.RotateTowards(lowerMuscle.targetRotation, Quaternion.Inverse(IkRelativeRotationFromStartLower), 1080f * Time.deltaTime);

            currentLowerRotationDifference = Quaternion.Angle(lowerMuscle.targetRotation, Quaternion.Inverse(IkRelativeRotationFromStartLower));
        }
    }
    // Update is called once per frame
    protected virtual void FixedUpdate()
    {

        {
            RunIkMuscleFollow();
        }


    }
}
