using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Abstract class for managing the transforms of a hierarchy of objects.
    /// This could be GameObjects or Monobehaviours in the scene graph, but this is
    /// not a requirement; for example, the objects could be rigid bodies whose hierarchy
    /// is defined by Joint configurations.
    ///
    /// Poses are either considered in model space, which is relative to a root body,
    /// or in local space, which is relative to their parent.
    /// </summary>
    public abstract class PoseExtractor
    {
        int[] m_ParentIndices;
        Pose[] m_ModelSpacePoses;
        Pose[] m_LocalSpacePoses;

        /// <summary>
        /// Read access to the model space transforms.
        /// </summary>
        public IList<Pose> ModelSpacePoses
        {
            get { return m_ModelSpacePoses;  }
        }

        /// <summary>
        /// Read access to the local space transforms.
        /// </summary>
        public IList<Pose> LocalSpacePoses
        {
            get { return m_LocalSpacePoses;  }
        }

        /// <summary>
        /// Number of transforms in the hierarchy (read-only).
        /// </summary>
        public int NumPoses
        {
            get { return m_ModelSpacePoses?.Length ?? 0;  }
        }

        /// <summary>
        /// Initialize with the mapping of parent indices.
        /// The 0th element is assumed to be -1, indicating that it's the root.
        /// </summary>
        /// <param name="parentIndices"></param>
        protected void SetParentIndices(int[] parentIndices)
        {
            m_ParentIndices = parentIndices;
            var numTransforms = parentIndices.Length;
            m_ModelSpacePoses = new Pose[numTransforms];
            m_LocalSpacePoses = new Pose[numTransforms];
        }

        /// <summary>
        /// Return the world space Pose of the i'th object.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        protected abstract Pose GetPoseAt(int index);

        /// <summary>
        /// Update the internal model space transform storage based on the underlying system.
        /// </summary>
        public void UpdateModelSpacePoses()
        {
            if (m_ModelSpacePoses == null)
            {
                return;
            }

            var worldTransform = GetPoseAt(0);
            var worldToModel = worldTransform.Inverse();

            for (var i = 0; i < m_ModelSpacePoses.Length; i++)
            {
                var currentTransform = GetPoseAt(i);
                m_ModelSpacePoses[i] = worldToModel.Multiply(currentTransform);
            }
        }

        /// <summary>
        /// Update the internal model space transform storage based on the underlying system.
        /// </summary>
        public void UpdateLocalSpacePoses()
        {
            if (m_LocalSpacePoses == null)
            {
                return;
            }

            for (var i = 0; i < m_LocalSpacePoses.Length; i++)
            {
                if (m_ParentIndices[i] != -1)
                {
                    var parentTransform = GetPoseAt(m_ParentIndices[i]);
                    // This is slightly inefficient, since for a body with multiple children, we'll end up inverting
                    // the transform multiple times. Might be able to trade space for perf here.
                    var invParent = parentTransform.Inverse();
                    var currentTransform = GetPoseAt(i);
                    m_LocalSpacePoses[i] = invParent.Multiply(currentTransform);
                }
                else
                {
                    m_LocalSpacePoses[i] = Pose.identity;
                }
            }
        }


        public void DrawModelSpace(Vector3 offset)
        {
            UpdateLocalSpacePoses();
            UpdateModelSpacePoses();

            var pose = m_ModelSpacePoses;
            var localPose = m_LocalSpacePoses;
            for (var i = 0; i < pose.Length; i++)
            {
                var current = pose[i];
                if (m_ParentIndices[i] == -1)
                {
                    continue;
                }

                var parent = pose[m_ParentIndices[i]];
                Debug.DrawLine(current.position + offset, parent.position + offset, Color.cyan);
                var localUp = localPose[i].rotation * Vector3.up;
                var localFwd = localPose[i].rotation * Vector3.forward;
                var localRight = localPose[i].rotation * Vector3.right;
                Debug.DrawLine(current.position+offset, current.position+offset+.1f*localUp, Color.red);
                Debug.DrawLine(current.position+offset, current.position+offset+.1f*localFwd, Color.green);
                Debug.DrawLine(current.position+offset, current.position+offset+.1f*localRight, Color.blue);
            }
        }
    }

    public static class PoseExtensions
    {
        /// <summary>
        /// Compute the inverse of a Pose. For any Pose P,
        ///   P.Inverse() * P
        /// will equal the identity pose (within tolerance).
        /// </summary>
        /// <param name="pose"></param>
        /// <returns></returns>
        public static Pose Inverse(this Pose pose)
        {
            var rotationInverse = Quaternion.Inverse(pose.rotation);
            var translationInverse = -(rotationInverse * pose.position);
            return new Pose { rotation = rotationInverse, position = translationInverse };
        }

        /// <summary>
        /// This is equivalent to Pose.GetTransformedBy(), but keeps the order more intuitive.
        /// </summary>
        /// <param name="pose"></param>
        /// <param name="rhs"></param>
        /// <returns></returns>
        public static Pose Multiply(this Pose pose, Pose rhs)
        {
            return rhs.GetTransformedBy(pose);
        }

        // TODO optimize inv(A)*B?
    }
}
