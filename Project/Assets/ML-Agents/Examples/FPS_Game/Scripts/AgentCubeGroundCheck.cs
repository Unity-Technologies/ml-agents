//Standardized ground check for the Agent Cube
using UnityEngine;

namespace MLAgents
{

    /// <summary>
    /// Perform Groundcheck using a Physics OverlapBox
    /// </summary>
    [DisallowMultipleComponent]
    public class AgentCubeGroundCheck : MonoBehaviour
    {
        public bool debugDrawGizmos;
        public Collider[] hitGroundColliders = new Collider[3];
        public Vector3 groundCheckBoxLocalPos = new Vector3(0, -0.52f, 0);
        public Vector3 groundCheckBoxSize = new Vector3(0.99f, 0.1f, 0.99f);
        public bool isGrounded;
        public float ungroundedTime; //amount of time agent hasn't been grounded
        public float groundedTime; //amount of time agent has been grounded

        void FixedUpdate()
        {
            DoGroundCheck();
            if (!isGrounded)
            {
                ungroundedTime += Time.deltaTime;
                groundedTime = 0;
            }
            else
            {
                groundedTime += Time.deltaTime;
                ungroundedTime = 0;
            }
        }

        /// <summary>
        /// Does the ground check.
        /// </summary>
        /// <returns><c>true</c>, if the agent is on the ground,
        /// <c>false</c> otherwise.</returns>
        /// <param name="smallCheck"></param>
        public void DoGroundCheck()
        {
            isGrounded = false;
            if (Physics.OverlapBoxNonAlloc(
                transform.TransformPoint(groundCheckBoxLocalPos),
                groundCheckBoxSize / 2,
                hitGroundColliders,
                transform.rotation) > 0)
            {
                foreach (var col in hitGroundColliders)
                {
                    if (col != null && col.transform != transform &&
                        (col.CompareTag("walkableSurface")
                         || col.CompareTag("ground")
                         || col.CompareTag("block")))
                    {
                        isGrounded = true; //then we're grounded
                        break;
                    }
                }
            }
            //empty the array
            for (int i = 0; i < hitGroundColliders.Length; i++)
            {
                hitGroundColliders[i] = null;
            }
        }

        //Draw the Box Overlap as a gizmo to show where it currently is testing.
        void OnDrawGizmos()
        {
            if (debugDrawGizmos)
            {
                Gizmos.color = Color.red;
                Gizmos.matrix = transform.localToWorldMatrix;
                Gizmos.DrawWireCube(groundCheckBoxLocalPos, groundCheckBoxSize);
            }
        }
    }
}
