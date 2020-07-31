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
        /// Initialize given a root RigidBody.
        /// </summary>
        /// <param name="rootBody"></param>
        public RigidBodyPoseExtractor(Rigidbody rootBody, GameObject rootGameObject = null)
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
            var bodyToIndex = new Dictionary<Rigidbody, int>(rbs.Length);
            var parentIndices = new int[rbs.Length];

            if (rbs[0] != rootBody)
            {
                Debug.Log("Expected root body at index 0");
                return;
            }

            for (var i = 0; i < rbs.Length; i++)
            {
                bodyToIndex[rbs[i]] = i;
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

            m_Bodies = rbs;
            SetParentIndices(parentIndices);
        }

        /// <inheritdoc/>
        protected override Vector3 GetLinearVelocityAt(int index)
        {
            return m_Bodies[index].velocity;
        }

        /// <inheritdoc/>
        protected override Pose GetPoseAt(int index)
        {
            var body = m_Bodies[index];
            return new Pose { rotation = body.rotation, position = body.position };
        }

        internal Rigidbody[] Bodies => m_Bodies;
    }

}
