using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Extensions.Sensors
{

    /// <summary>
    /// Utility class to track a hierarchy of RigidBodies. These are assumed to have a root node,
    /// and child nodes are connect to their parents via Joints.
    /// </summary>
    public class RigidBodyHierarchyUtil : HierarchyUtil
    {
        Rigidbody[] m_Bodies;

        public RigidBodyHierarchyUtil(Rigidbody rootBody)
        {
            // TODO pass root body, walk constraint chain for each body until reach root or parented body
            var rbs = rootBody.GetComponentsInChildren <Rigidbody>();
            var joints = rootBody.GetComponentsInChildren <Joint>();

            var parentMap = new Dictionary<Rigidbody, Rigidbody>();
            foreach (var rb in rbs)
            {
                parentMap[rb] = null;
            }

            foreach (var j in joints)
            {
                var parent = j.connectedBody;
                var child = j.GetComponent<Rigidbody>();
                parentMap[child] = parent;
            }

            foreach (var pair in parentMap)
            {
                if (pair.Value == null && pair.Key != rootBody)
                {
                    Debug.Log($"Found body {pair.Key} with no parent. exiting");
                    return;
                }
            }

            m_Bodies = new Rigidbody[rbs.Length];
            var parentIndices = new int[rbs.Length];
            var bodyToIndex = new Dictionary<Rigidbody, int>(rbs.Length);

            m_Bodies[0] = rootBody;
            parentIndices[0] = -1;
            bodyToIndex[rootBody] = 0;
            var index = 1;

            // This is inefficient in the worst case (e.g. a chain)
            // And might not terminate?
            while(bodyToIndex.Count != rbs.Length)
            {
                foreach (var rb in rbs)
                {
                    if (bodyToIndex.ContainsKey(rb))
                    {
                        // Already found a place for this one
                        continue;
                    }

                    var parent = parentMap[rb];
                    if (bodyToIndex.ContainsKey(parent))
                    {
                        m_Bodies[index] = rb;
                        parentIndices[index] = bodyToIndex[parent];
                        bodyToIndex[rb] = index;
                        index++;
                    }
                }
            }

            SetParentIndices(parentIndices);
        }

        protected override QTTransform GetTransformAt(int index)
        {
            var body = m_Bodies[index];
            return new QTTransform { Rotation = body.rotation, Translation = body.position };
        }


    }
}
