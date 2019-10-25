using System;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Ray perception component. Attach this to agents to enable "local perception"
    /// via the use of ray casts directed outward from the agent.
    /// </summary>
    public class RayPerception3D : RayPerception
    {
        Vector3 m_EndPosition;
        RaycastHit m_Hit;
        private float[] m_SubList;

        /// <summary>
        /// Creates perception vector to be used as part of an observation of an agent.
        /// Each ray in the rayAngles array adds a sublist of data to the observation.
        /// The sublist contains the observation data for a single ray. The list is composed of the following:
        /// 1. A one-hot encoding for detectable objects. For example, if detectableObjects.Length = n, the
        ///    first n elements of the sublist will be a one-hot encoding of the detectableObject that was hit, or
        ///    all zeroes otherwise.
        /// 2. The 'length' element of the sublist will be 1 if the ray missed everything, or 0 if it hit
        ///    something (detectable or not).
        /// 3. The 'length+1' element of the sublist will contain the normalised distance to the object hit.
        /// NOTE: Only objects with tags in the detectableObjects array will have a distance set.
        /// </summary>
        /// <returns>The partial vector observation corresponding to the set of rays</returns>
        /// <param name="rayDistance">Radius of rays</param>
        /// <param name="rayAngles">Angles of rays (starting from (1,0) on unit circle).</param>
        /// <param name="detectableObjects">List of tags which correspond to object types agent can see</param>
        /// <param name="startOffset">Starting height offset of ray from center of agent.</param>
        /// <param name="endOffset">Ending height offset of ray from center of agent.</param>
        public override List<float> Perceive(float rayDistance,
            float[] rayAngles, string[] detectableObjects,
            float startOffset=0.0f, float endOffset=0.0f)
        {
            if (m_SubList == null || m_SubList.Length != detectableObjects.Length + 2)
                m_SubList = new float[detectableObjects.Length + 2];

            m_PerceptionBuffer.Clear();
            m_PerceptionBuffer.Capacity = m_SubList.Length * rayAngles.Length;

            // For each ray sublist stores categorical information on detected object
            // along with object distance.
            foreach (var angle in rayAngles)
            {
                m_EndPosition = transform.TransformDirection(
                    PolarToCartesian(rayDistance, angle));
                m_EndPosition.y = endOffset;
                if (Application.isEditor)
                {
                    Debug.DrawRay(transform.position + new Vector3(0f, startOffset, 0f),
                        m_EndPosition, Color.black, 0.01f, true);
                }

                Array.Clear(m_SubList, 0, m_SubList.Length);

                if (Physics.SphereCast(transform.position +
                    new Vector3(0f, startOffset, 0f), 0.5f,
                    m_EndPosition, out m_Hit, rayDistance))
                {
                    for (var i = 0; i < detectableObjects.Length; i++)
                    {
                        if (m_Hit.collider.gameObject.CompareTag(detectableObjects[i]))
                        {
                            m_SubList[i] = 1;
                            m_SubList[detectableObjects.Length + 1] = m_Hit.distance / rayDistance;
                            break;
                        }
                    }
                }
                else
                {
                    m_SubList[detectableObjects.Length] = 1f;
                }

                Utilities.AddRangeNoAlloc(m_PerceptionBuffer, m_SubList);
            }

            return m_PerceptionBuffer;
        }

        /// <summary>
        /// Converts polar coordinate to cartesian coordinate.
        /// </summary>
        public static Vector3 PolarToCartesian(float radius, float angle)
        {
            var x = radius * Mathf.Cos(DegreeToRadian(angle));
            var z = radius * Mathf.Sin(DegreeToRadian(angle));
            return new Vector3(x, 0f, z);
        }
    }
}
