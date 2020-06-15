using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Simple linear transform representation consisting of a rotation and translation.
    /// When deriving the math, it can be helpful to think of the transform as a 4x4 block matrix:
    /// <example>
    /// | R | t |
    /// ----+----
    /// | 0 | 1 |
    /// </example>
    /// where R is a 3x3 rotation, t is a 3x1 translation, 0 is a 1x3 vector of 0s.
    /// </summary>
    public struct QTTransform
    {
        public Quaternion Rotation;
        public Vector3 Translation;

        /// <summary>
        /// Multiply two transforms.
        /// <example>
        /// | R1 | t1 |    | R2 | t2 |   | R1*R2 | R1*t2 + t1 |
        /// -----+----- *  -----+----- = --------+-------------
        /// | 0  | 1  |    | 0  | 1  |   |   0   |      1     |
        /// </example>
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <returns></returns>
        public static QTTransform operator *(QTTransform t1, QTTransform t2)
        {
            var translation = (t1.Rotation * t2.Translation) + t1.Translation;
            var rotation = t1.Rotation * t2.Rotation;
            return new QTTransform { Rotation = rotation, Translation = translation };
        }

        public QTTransform Inverse()
        {
            var rotationInverse = Quaternion.Inverse(Rotation);
            var translationInverse = -(rotationInverse * Translation);
            return new QTTransform { Rotation = rotationInverse, Translation = translationInverse };
        }

        public static QTTransform Identity
        {
            get { return new QTTransform { Rotation = Quaternion.identity, Translation = Vector3.zero}; }
        }

        // TODO optimize inv(A)*B?

    }

    public abstract class HierarchyUtil
    {

//        protected SetNumParentIndices(int[] parentIndices)
//        {
//
//        }

    }

    public class RigidBodyHierarchyUtil
    {
        Rigidbody[] m_Bodies;
        int[] m_ParentIndices;
        Dictionary<Rigidbody, int> m_BodyToIndex;

        public void InitTree(GameObject sourceBody)
        {
            // TODO pass root body, walk constraint chain for each body until reach root or parented body
            var rbs = sourceBody.GetComponentsInChildren <Rigidbody>();
            var joints = sourceBody.GetComponentsInChildren <Joint>();

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

            int numRoots = 0;
            Rigidbody root = null;
            foreach (var pair in parentMap)
            {
                if (pair.Value == null)
                {
                    numRoots++;
                    root = pair.Key;
                }
            }

            // Hopefully exactly one root
            if (numRoots != 1)
            {
                Debug.Log($"Found {numRoots} roots. exiting");
                return;
            }

            m_Bodies = new Rigidbody[rbs.Length];
            m_ParentIndices = new int[rbs.Length];
            m_BodyToIndex = new Dictionary<Rigidbody, int>(rbs.Length);

            m_Bodies[0] = root;
            m_ParentIndices[0] = -1;
            m_BodyToIndex[root] = 0;
            var index = 1;

            // This is inefficient in the worst case (e.g. a chain)
            // And might not terminate?
            while(m_BodyToIndex.Count != rbs.Length)
            {
                foreach (var rb in rbs)
                {
                    if (m_BodyToIndex.ContainsKey(rb))
                    {
                        continue;
                    }

                    var parent = parentMap[rb];
                    if (m_BodyToIndex.ContainsKey(parent))
                    {
                        m_Bodies[index] = rb;
                        m_ParentIndices[index] = m_BodyToIndex[parent];
                        m_BodyToIndex[rb] = index;
                        index++;
                    }
                }
            }

        }

        public QTTransform[] GetModelSpaceTransforms()
        {
            var rootRb = m_Bodies[0];
            var worldTransform = new QTTransform { Rotation = rootRb.rotation, Translation = rootRb.position };
            var worldToModel = worldTransform.Inverse();

            var output = new QTTransform[m_Bodies.Length];
            for (var i = 0; i < m_Bodies.Length; i++)
            {
                var body = m_Bodies[i];
                var rbTransform = new QTTransform { Rotation = body.rotation, Translation = body.position };
                output[i] = worldToModel * rbTransform;
            }

            return output;
        }

        public QTTransform[] GetLocalSpaceTransforms()
        {
            var output = new QTTransform[m_Bodies.Length];
            for (var i = 0; i < m_Bodies.Length; i++)
            {
                var body = m_Bodies[i];
                if (m_ParentIndices[i] != -1)
                {
                    var parentBody = m_Bodies[m_ParentIndices[i]];
                    var parentTransform = new QTTransform { Rotation = parentBody.rotation, Translation = parentBody.position };
                    // This is slightly inefficient, since for a body with multiple children, we'll end up inverting
                    // the transform multiple times. Might be able to trade space for perf here.
                    var invParent = parentTransform.Inverse();
                    var rbTransform = new QTTransform { Rotation = body.rotation, Translation = body.position };
                    output[i] = invParent * rbTransform;
                }
                else
                {
                    output[i] = QTTransform.Identity;
                }
            }

            return output;
        }

        public void DrawModelSpace(Vector3 offset, Color color)
        {
            var pose = GetModelSpaceTransforms();
            var localPose = GetLocalSpaceTransforms();
            for (var i = 0; i < pose.Length; i++)
            {
                var current = pose[i];
                if (m_ParentIndices[i] == -1)
                {
                    continue;
                }

                var parent = pose[m_ParentIndices[i]];
                Debug.DrawLine(current.Translation+offset, parent.Translation+offset, Color.cyan);
                var localUp = localPose[i].Rotation * Vector3.up;
                var localFwd = localPose[i].Rotation * Vector3.forward;
                var localRight = localPose[i].Rotation * Vector3.right;
                Debug.DrawLine(current.Translation+offset, current.Translation+offset+.1f*localUp, Color.red);
                Debug.DrawLine(current.Translation+offset, current.Translation+offset+.1f*localFwd, Color.green);
                Debug.DrawLine(current.Translation+offset, current.Translation+offset+.1f*localRight, Color.blue);
            }
        }
    }
}
