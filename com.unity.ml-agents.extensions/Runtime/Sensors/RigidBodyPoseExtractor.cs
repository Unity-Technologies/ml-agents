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

        internal override List<TreeNode> GetTreeNodes()
        {
            var nodesOut = new List<TreeNode>(m_Bodies.Length);
//            for (var i = 0; i < m_Bodies.Length; i++)
//            {
//                Object obj = null;
//                if (i == 0 && m_VirtualRoot != null)
//                {
//                    obj = m_VirtualRoot;
//                }
//                else
//                {
//                    obj = m_Bodies[i];
//                }
//                var node = new TreeNode
//                {
//                    NodeObject = obj,
//                    Enabled = true
//                };
//                nodesOut.Add(node);
//            }

            // List of children for each node
            var tree = new Dictionary<int, List<int>>();
            for (var i = 0; i < m_Bodies.Length; i++)
            {
                var parent = GetParentIndex(i);
                if (i == -1)
                {
                    continue;
                }

                if (!tree.ContainsKey(parent))
                {
                    tree[parent] = new List<int>();
                }
                tree[parent].Add(i);
            }

            // Store (index, depth) in the stack
            var stack = new Stack<(int, int)>();
            stack.Push((0, 0));

            while (stack.Count != 0)
            {
                var (current, depth) = stack.Pop();
                Object obj = null;
                if (current == 0 && m_VirtualRoot != null)
                {
                    obj = m_VirtualRoot;
                }
                else
                {
                    obj = m_Bodies[current];
                }

                var node = new TreeNode
                {
                    NodeObject = obj,
                    Enabled = true,
                    OriginalIndex = current,
                    Depth = depth
                };
                nodesOut.Add(node);

                // Add children
                // TOOD check for already visited. Shouldn't happen, but we'd blow up on loops.
                if (tree.ContainsKey(current))
                {
                    // Push to the stack in reverse order
                    var children = tree[current];
                    for (var childIdx = children.Count-1; childIdx >= 0; childIdx--)
                    {
                        stack.Push((children[childIdx], depth+1));
                    }
                }

                // Safety check
                {
                    if (nodesOut.Count > m_Bodies.Length)
                    {
                        Debug.Log("oops");
                        return nodesOut;
                    }
                }
            }

            return nodesOut;
        }
    }

}
