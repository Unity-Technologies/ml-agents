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
        /// a stabilized reference frame, which can improve learning.</param>
        /// <param name="enableBodyPoses">Optional mapping of whether a body's psoe should be enabled or not.</param>
        public RigidBodyPoseExtractor(Rigidbody rootBody, GameObject rootGameObject = null,
            GameObject virtualRoot = null, Dictionary<Rigidbody, bool> enableBodyPoses = null)
        {
            if (rootBody == null)
            {
                return;
            }

            Rigidbody[] rbs;
            Joint[] joints;
            if (rootGameObject == null)
            {
                rbs = rootBody.GetComponentsInChildren<Rigidbody>();
                joints = rootBody.GetComponentsInChildren<Joint>();
            }
            else
            {
                rbs = rootGameObject.GetComponentsInChildren<Rigidbody>();
                joints = rootGameObject.GetComponentsInChildren<Joint>();
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
                if (rbs[i] != null)
                {
                    bodyToIndex[rbs[i]] = i;
                }
            }

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

            if (enableBodyPoses != null)
            {
                foreach (var pair in enableBodyPoses)
                {
                    var rb = pair.Key;
                    if (bodyToIndex.TryGetValue(rb, out var index))
                    {
                        SetPoseEnabled(index, pair.Value);
                    }
                }
            }
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

        /// <inheritdoc/>
        protected internal override Object GetObjectAt(int index)
        {
            if (index == 0 && m_VirtualRoot != null)
            {
                return m_VirtualRoot;
            }
            return m_Bodies[index];
        }

        internal Rigidbody[] Bodies => m_Bodies;

        /// <summary>
        /// Get a dictionary indicating which Rigidbodies' poses are enabled or disabled.
        /// </summary>
        /// <returns></returns>
        internal Dictionary<Rigidbody, bool> GetBodyPosesEnabled()
        {
            var bodyPosesEnabled = new Dictionary<Rigidbody, bool>(m_Bodies.Length);
            for (var i = 0; i < m_Bodies.Length; i++)
            {
                var rb = m_Bodies[i];
                if (rb == null)
                {
                    continue; // skip virtual root
                }

                bodyPosesEnabled[rb] = IsPoseEnabled(i);
            }

            return bodyPosesEnabled;
        }

        internal IEnumerable<Rigidbody> GetEnabledRigidbodies()
        {
            if (m_Bodies == null)
            {
                yield break;
            }

            for (var i = 0; i < m_Bodies.Length; i++)
            {
                var rb = m_Bodies[i];
                if (rb == null)
                {
                    // Ignore a virtual root.
                    continue;
                }

                if (IsPoseEnabled(i))
                {
                    yield return rb;
                }
            }
        }
    }

}
