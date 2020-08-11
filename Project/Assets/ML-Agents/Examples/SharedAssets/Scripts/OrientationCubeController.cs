using UnityEngine;

namespace Unity.MLAgentsExamples
{
    /// <summary>
    /// Utility class to allow a stable observation platform.
    /// </summary>
    public class OrientationCubeController : MonoBehaviour
    {
        //Public method to allow Agent to set look rotation of this transform
        public void UpdateOrientation(Transform rootBP, Transform target)
        {
            var dirVector = target.position - transform.position;
            dirVector.y = 0; //flatten dir on the y. this will only work on level surfaces
            var lookRot =
                dirVector == Vector3.zero
                    ? Quaternion.identity
                    : Quaternion.LookRotation(dirVector); //get our look rot to the target

            //UPDATE ORIENTATION CUBE POS & ROT
            transform.SetPositionAndRotation(rootBP.position, lookRot);
        }
    }
}
