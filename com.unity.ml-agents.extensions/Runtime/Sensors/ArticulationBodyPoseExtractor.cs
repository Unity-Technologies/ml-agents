#if UNITY_2020_1_OR_NEWER

using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Utility class to track a hierarchy of ArticulationBodies.
    /// </summary>
    public class ArticulationBodyPoseExtractor : PoseExtractor
    {
        ArticulationBody[] m_Bodies;

        public ArticulationBodyPoseExtractor(ArticulationBody rootBody)
        {
            if (rootBody == null)
            {
                return;
            }

            if (!rootBody.isRoot)
            {
                Debug.Log("Must pass ArticulationBody.isRoot");
                return;
            }

            var bodies = rootBody.GetComponentsInChildren<ArticulationBody>();
            if (bodies[0] != rootBody)
            {
                Debug.Log("Expected root body at index 0");
                return;
            }

            var numBodies = bodies.Length;
            m_Bodies = bodies;
            int[] parentIndices = new int[numBodies];
            parentIndices[0] = -1;

            var bodyToIndex = new Dictionary<ArticulationBody, int>();
            for (var i = 0; i < numBodies; i++)
            {
                bodyToIndex[m_Bodies[i]] = i;
            }

            for (var i = 1; i < numBodies; i++)
            {
                var currentArticBody = m_Bodies[i];
                // Component.GetComponentInParent will consider the provided object as well.
                // So start looking from the parent.
                var currentGameObject = currentArticBody.gameObject;
                var parentGameObject = currentGameObject.transform.parent;
                var parentArticBody = parentGameObject.GetComponentInParent<ArticulationBody>();
                parentIndices[i] = bodyToIndex[parentArticBody];
            }

            Setup(parentIndices);
        }

        /// <inheritdoc/>
        protected internal override Vector3 GetLinearVelocityAt(int index)
        {
            return m_Bodies[index].velocity;
        }

        /// <inheritdoc/>
        protected internal override Pose GetPoseAt(int index)
        {
            var body = m_Bodies[index];
            var go = body.gameObject;
            var t = go.transform;
            return new Pose { rotation = t.rotation, position = t.position };
        }

        /// <inheritdoc/>
        protected internal override Object GetObjectAt(int index)
        {
            return m_Bodies[index];
        }

        internal ArticulationBody[] Bodies => m_Bodies;

        internal IEnumerable<ArticulationBody> GetEnabledArticulationBodies()
        {
            if (m_Bodies == null)
            {
                yield break;
            }

            for (var i = 0; i < m_Bodies.Length; i++)
            {
                var articBody = m_Bodies[i];
                if (articBody == null)
                {
                    // Ignore a virtual root.
                    continue;
                }

                if (IsPoseEnabled(i))
                {
                    yield return articBody;
                }
            }
        }
    }
}
#endif // UNITY_2020_1_OR_NEWER
