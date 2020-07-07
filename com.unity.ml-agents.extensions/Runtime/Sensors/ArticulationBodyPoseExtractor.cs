#if UNITY_2020_1_OR_NEWER

using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Extensions.Sensors
{

    public class ArticulationBodyPoseExtractor : PoseExtractor
    {
        ArticulationBody[] m_Bodies;

        public ArticulationBodyPoseExtractor(ArticulationBody rootBody)
        {
            if (!rootBody.isRoot)
            {
                Debug.Log("Must pass ArticulationBody.isRoot");
                return;
            }

            var bodies = rootBody.GetComponentsInChildren <ArticulationBody>();
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
                var body = m_Bodies[i];
                var parent = body.GetComponentInParent<ArticulationBody>();
                parentIndices[i] = bodyToIndex[parent];
            }

            SetParentIndices(parentIndices);
        }

        protected override Pose GetPoseAt(int index)
        {
            var body = m_Bodies[index];
            var go = body.gameObject;
            var t = go.transform;
            return new Pose { rotation = t.rotation, position = t.position };
        }


    }
}
#endif // UNITY_2020_1_OR_NEWER