using UnityEngine;

namespace Unity.MLAgentsExamples
{
    /// <summary>
    /// Utility class to allow a stable observation platform.
    /// </summary>
    public class SetLocalUpToWorldUp : MonoBehaviour
    {
        public Transform AttachedToTransform;//the transform we want to match rotation for
        // //Update position and Rotation
        // public void UpdateOrientation(Transform rootBP, Transform target)
        // {
        //     var dirVector = target.position - transform.position;
        //     dirVector.y = 0; //flatten dir on the y. this will only work on level, uneven surfaces
        //     var lookRot =
        //         dirVector == Vector3.zero
        //             ? Quaternion.identity
        //             : Quaternion.LookRotation(dirVector); //get our look rot to the target
        //
        //     //UPDATE ORIENTATION CUBE POS & ROT
        //     transform.SetPositionAndRotation(rootBP.position, lookRot);
        // }

        void FixedUpdate()
        {
            var dirVector = AttachedToTransform.forward;
            dirVector.y = 0; //flatten dir on the y. this will only work on level, uneven surfaces
            var lookRot = Quaternion.LookRotation(dirVector); //get our look rot to the target

            //UPDATE ORIENTATION CUBE POS & ROT
            transform.SetPositionAndRotation(AttachedToTransform.position, lookRot);

            // var currentRot = transform.rotation.eulerAngles;
            // currentRot.y = 0;
            // transform.rotation = Quaternion.Euler(currentRot);
        }
    }

}
