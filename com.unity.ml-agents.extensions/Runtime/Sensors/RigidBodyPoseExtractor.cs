using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Extensions.Sensors
{

    /// <summary>
    /// Utility class to track a hierarchy of RigidBodies. These are assumed to have a root node,
    /// and child nodes are connect to their parents via Joints.
    /// </summary>
    public class RigidBodyPoseExtractor : PoseExtractor
    {
        Rigidbody[] m_Bodies;

        /// <summary>
        /// Optional game object used to determine the root of the poses, separate from the actual Rigidbodies
        /// in the hierarchy. For locomotion
        /// </summary>
        GameObject m_VirtualRoot;

        /// <summary>
        /// Initialize given a root RigidBody.
        /// </summary>
        /// <param name="rootBody">The root Rigidbody. This has no Joints on it (but other Joints may connect to it).</param>
        /// <param name="rootGameObject">Optional GameObject used to find Rigidbodies in the hierarchy.</param>
        /// <param name="virtualRoot">Optional GameObject used to determine the root of the poses,
        /// separate from the actual Rigidbodies in the hierarchy. For locomotion tasks, with ragdolls, this provides
        /// a stabilized refernece frame, which can improve learning.</param>
        public RigidBodyPoseExtractor(Rigidbody rootBody, GameObject rootGameObject = null, GameObject virtualRoot = null)
        {
            if (rootBody == null)
            {
                return;
            }

            Rigidbody[] rbs;
            if (rootGameObject == null)
            {
                rbs = rootBody.GetComponentsInChildren<Rigidbody>();
            }
            else
            {
                rbs = rootGameObject.GetComponentsInChildren<Rigidbody>();
            }

            if (rbs == null || rbs.Length == 0)
            {
                Debug.Log("No rigid bodies found!");
                return;
            }

                if (rbs[0] != rootBody)
            {
                Debug.Log("Expected root body at index 0");
                return;
            }

            // Adjust the array if we have a virtual root.
            // This will be at index 0, and the "real" root will be parented to it.
            if (virtualRoot != null)
            {
                var extendedRbs = new Rigidbody[rbs.Length + 1];
                for (var i = 0; i < rbs.Length; i++)
                {
                    extendedRbs[i + 1] = rbs[i];
                }

                rbs = extendedRbs;
            }

            var bodyToIndex = new Dictionary<Rigidbody, int>(rbs.Length);
            var parentIndices = new int[rbs.Length];
            parentIndices[0] = -1;

            for (var i = 0; i < rbs.Length; i++)
            {
                if(rbs[i] != null)
                {
                    bodyToIndex[rbs[i]] = i;
                }
            }

            var joints = rootBody.GetComponentsInChildren <Joint>();


            foreach (var j in joints)
            {
                var parent = j.connectedBody;
                var child = j.GetComponent<Rigidbody>();

                var parentIndex = bodyToIndex[parent];
                var childIndex = bodyToIndex[child];
                parentIndices[childIndex] = parentIndex;
            }

            if (virtualRoot != null)
            {
                // Make sure the original root treats the virtual root as its parent.
                parentIndices[1] = 0;
                m_VirtualRoot = virtualRoot;
            }

            m_Bodies = rbs;
            Setup(parentIndices);

            // By default, ignore the root
            SetPoseEnabled(0, false);
        }

        /// <inheritdoc/>
        protected internal override Vector3 GetLinearVelocityAt(int index)
        {
            if (index == 0 && m_VirtualRoot != null)
            {
                // No velocity on the virtual root
                return Vector3.zero;
            }
            return m_Bodies[index].velocity;
        }

        /// <inheritdoc/>
        protected internal override Pose GetPoseAt(int index)
        {
            if (index == 0 && m_VirtualRoot != null)
            {
                // Use the GameObject's world transform
                return new Pose
                {
                    rotation = m_VirtualRoot.transform.rotation,
                    position = m_VirtualRoot.transform.position
                };
            }

            var body = m_Bodies[index];
            return new Pose { rotation = body.rotation, position = body.position };
        }

        internal Rigidbody[] Bodies => m_Bodies;
    }

}
