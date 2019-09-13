using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Ray 2D perception component. Attach this to agents to enable "local perception"
    /// via the use of ray casts directed outward from the agent.
    /// </summary>
    public class RayPerception2D : RayPerception
    {
        Vector2 m_EndPosition;
        RaycastHit2D m_Hit;

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
        public List<float> Perceive(float rayDistance,
            float[] rayAngles, string[] detectableObjects)
        {
            m_PerceptionBuffer.Clear();
            // For each ray sublist stores categorical information on detected object
            // along with object distance.
            foreach (var angle in rayAngles)
            {
                m_EndPosition = transform.TransformDirection(
                    PolarToCartesian(rayDistance, angle));
                if (Application.isEditor)
                {
                    Debug.DrawRay(transform.position,
                        m_EndPosition, Color.black, 0.01f, true);
                }

                var subList = new float[detectableObjects.Length + 2];
                m_Hit = Physics2D.CircleCast(transform.position, 0.5f, m_EndPosition, rayDistance);
                if (m_Hit)
                {
                    for (var i = 0; i < detectableObjects.Length; i++)
                    {
                        if (m_Hit.collider.gameObject.CompareTag(detectableObjects[i]))
                        {
                            subList[i] = 1;
                            subList[detectableObjects.Length + 1] = m_Hit.distance / rayDistance;
                            break;
                        }
                    }
                }
                else
                {
                    subList[detectableObjects.Length] = 1f;
                }

                m_PerceptionBuffer.AddRange(subList);
            }

            return m_PerceptionBuffer;
        }

        /// <summary>
        /// Converts polar coordinate to cartesian coordinate.
        /// </summary>
        public static Vector2 PolarToCartesian(float radius, float angle)
        {
            var x = radius * Mathf.Cos(DegreeToRadian(angle));
            var y = radius * Mathf.Sin(DegreeToRadian(angle));
            return new Vector2(x, y);
        }
    }
}
