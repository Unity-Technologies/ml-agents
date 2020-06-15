using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Extensions.Sensors
{
    public abstract class HierarchyUtil
    {
        int[] m_ParentIndices;
        QTTransform[] m_ModelSpacePose;
        QTTransform[] m_LocalSpacePose;

        public IList<QTTransform> ModelSpacePose
        {
            get { return m_ModelSpacePose;  }
        }

        public IList<QTTransform> LocalSpacePose
        {
            get { return m_LocalSpacePose;  }
        }

        protected void SetParentIndices(int[] parentIndices)
        {
            m_ParentIndices = parentIndices;
            var numTransforms = parentIndices.Length;
            m_ModelSpacePose = new QTTransform[numTransforms];
            m_LocalSpacePose = new QTTransform[numTransforms];
        }

        protected abstract QTTransform GetTransformAt(int index);

        void UpdateModelSpaceTransforms()
        {
            var worldTransform = GetTransformAt(0);
            var worldToModel = worldTransform.Inverse();

            for (var i = 0; i < m_ModelSpacePose.Length; i++)
            {
                var currentTransform = GetTransformAt(i);
                m_ModelSpacePose[i] = worldToModel * currentTransform;
            }
        }

        void UpdateLocalSpaceTransforms()
        {
            for (var i = 0; i < m_LocalSpacePose.Length; i++)
            {
                if (m_ParentIndices[i] != -1)
                {
                    var parentTransform = GetTransformAt(m_ParentIndices[i]);
                    // This is slightly inefficient, since for a body with multiple children, we'll end up inverting
                    // the transform multiple times. Might be able to trade space for perf here.
                    var invParent = parentTransform.Inverse();
                    var currentTransform = GetTransformAt(i);
                    m_LocalSpacePose[i] = invParent * currentTransform;
                }
                else
                {
                    m_LocalSpacePose[i] = QTTransform.Identity;
                }
            }
        }

        public void DrawModelSpace(Vector3 offset)
        {
            UpdateLocalSpaceTransforms();
            UpdateModelSpaceTransforms();

            var pose = m_ModelSpacePose;
            var localPose = m_LocalSpacePose;
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
