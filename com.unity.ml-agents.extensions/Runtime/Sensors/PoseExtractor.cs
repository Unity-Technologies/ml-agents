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

        Vector3[] m_ModelSpaceLinearVelocities;
        Vector3[] m_LocalSpaceLinearVelocities;


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
        /// Read access to the model space linear velocities.
        /// </summary>
        public IList<Vector3> ModelSpaceVelocities
        {
            get { return m_ModelSpaceLinearVelocities;  }
        }

        /// <summary>
        /// Read access to the local space linear velocities.
        /// </summary>
        public IList<Vector3> LocalSpaceVelocities
        {
            get { return m_LocalSpaceLinearVelocities;  }
        }

        /// <summary>
        /// Number of poses in the hierarchy (read-only).
        /// </summary>
        public int NumPoses
        {
            get { return m_ModelSpacePoses?.Length ?? 0;  }
        }

        /// <summary>
        /// Get the parent index of the body at the specified index.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public int GetParentIndex(int index)
        {
            if (m_ParentIndices == null)
            {
                return -1;
            }

            return m_ParentIndices[index];
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

            m_ModelSpaceLinearVelocities = new Vector3[numTransforms];
            m_LocalSpaceLinearVelocities = new Vector3[numTransforms];
        }

        /// <summary>
        /// Return the world space Pose of the i'th object.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        protected abstract Pose GetPoseAt(int index);

        /// <summary>
        /// Return the world space linear velocity of the i'th object.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        protected abstract Vector3 GetLinearVelocityAt(int index);


        /// <summary>
        /// Update the internal model space transform storage based on the underlying system.
        /// </summary>
        public void UpdateModelSpacePoses()
        {
            if (m_ModelSpacePoses == null)
            {
                return;
            }

            var rootWorldTransform = GetPoseAt(0);
            var worldToModel = rootWorldTransform.Inverse();
            var rootLinearVel = GetLinearVelocityAt(0);

            for (var i = 0; i < m_ModelSpacePoses.Length; i++)
            {
                var currentWorldSpacePose = GetPoseAt(i);
                var currentModelSpacePose = worldToModel.Multiply(currentWorldSpacePose);
                m_ModelSpacePoses[i] = currentModelSpacePose;

                var currentBodyLinearVel = GetLinearVelocityAt(i);
                var relativeVelocity = currentBodyLinearVel - rootLinearVel;
                m_ModelSpaceLinearVelocities[i] = worldToModel.rotation * relativeVelocity;
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

                    var parentLinearVel = GetLinearVelocityAt(m_ParentIndices[i]);
                    var currentLinearVel = GetLinearVelocityAt(i);
                    m_LocalSpaceLinearVelocities[i] = invParent.rotation * (currentLinearVel - parentLinearVel);
                }
                else
                {
                    m_LocalSpacePoses[i] = Pose.identity;
                    m_LocalSpaceLinearVelocities[i] = Vector3.zero;
                }
            }
        }

        /// <summary>
        /// Compute the number of floats needed to represent the poses for the given PhysicsSensorSettings.
        /// </summary>
        /// <param name="settings"></param>
        /// <returns></returns>
        public int GetNumPoseObservations(PhysicsSensorSettings settings)
        {
            int obsPerPose = 0;
            obsPerPose += settings.UseModelSpaceTranslations ? 3 : 0;
            obsPerPose += settings.UseModelSpaceRotations ? 4 : 0;
            obsPerPose += settings.UseLocalSpaceTranslations ? 3 : 0;
            obsPerPose += settings.UseLocalSpaceRotations ? 4 : 0;

            obsPerPose += settings.UseModelSpaceLinearVelocity ? 3 : 0;
            obsPerPose += settings.UseLocalSpaceLinearVelocity ? 3 : 0;

            return NumPoses * obsPerPose;
        }

        internal void DrawModelSpace(Vector3 offset)
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

    /// <summary>
    /// Extension methods for the Pose struct, in order to improve the readability of some math.
    /// </summary>
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

        /// <summary>
        /// Transform the vector by the pose. Conceptually this is equivalent to treating the Pose
        /// as a 4x4 matrix and multiplying the augmented vector.
        /// See https://en.wikipedia.org/wiki/Affine_transformation#Augmented_matrix for more details.
        /// </summary>
        /// <param name="pose"></param>
        /// <param name="rhs"></param>
        /// <returns></returns>
        public static Vector3 Multiply(this Pose pose, Vector3 rhs)
        {
            return pose.rotation * rhs + pose.position;
        }

        // TODO optimize inv(A)*B?
    }
}
