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
        RaycastHit m_Hit;

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
        public override IList<float> Perceive(float rayDistance,
            float[] rayAngles, string[] detectableObjects,
            float startOffset=0.0f, float endOffset=0.0f)
        {
            var perceptionSize = (detectableObjects.Length + 2) * rayAngles.Length;
            if (m_PerceptionBuffer == null || m_PerceptionBuffer.Length != perceptionSize)
            {
                m_PerceptionBuffer = new float[perceptionSize];
            }

            PerceiveStatic(
                rayDistance, rayAngles, detectableObjects, startOffset, endOffset,
                transform, m_Hit, m_PerceptionBuffer
            );

            return m_PerceptionBuffer;
        }

        public static void PerceiveStatic(float rayDistance,
            IReadOnlyList<float> rayAngles, IReadOnlyList<string> detectableObjects,
            float startOffset, float endOffset, Transform transform, RaycastHit rayHit, float[] perceptionBuffer)
        {
            Array.Clear(perceptionBuffer, 0, perceptionBuffer.Length);

            // For each ray sublist stores categorical information on detected object
            // along with object distance.
            int bufferOffset = 0;
            foreach (var angle in rayAngles)
            {
                Vector3 startPositionLocal = new Vector3(0, startOffset, 0);
                Vector3 endPositionLocal = PolarToCartesian(rayDistance, angle);
                endPositionLocal.y += endOffset;

                var startPositionWorld = transform.TransformPoint(startPositionLocal);
                var endPositionWorld = transform.TransformPoint(endPositionLocal);

                var rayDirection = endPositionWorld - startPositionWorld;

                // Do the raycast and assign the hit information for each detectable object.
                //   sublist[0           ] <- did hit detectableObjects[0]
                //   ...
                //   sublist[numObjects-1] <- did hit detectableObjects[numObjects-1]
                //   sublist[numObjects  ] <- 1 if missed else 0
                //   sublist[numObjects+1] <- hit fraction if hit else 0
                // Note that if the raycast hits a object that's not in the detectableObjects list, all results are 0

                if (Physics.SphereCast(startPositionWorld, 0.5f, rayDirection, out rayHit, rayDistance))
                {
                    for (var i = 0; i < detectableObjects.Count; i++)
                    {
                        if (rayHit.collider.gameObject.CompareTag(detectableObjects[i]))
                        {
                            perceptionBuffer[bufferOffset + i] = 1;
                            perceptionBuffer[bufferOffset + detectableObjects.Count + 1] = rayHit.distance / rayDistance;
                            break;
                        }
                    }
                }
                else
                {
                    perceptionBuffer[bufferOffset +detectableObjects.Count] = 1f;
                }

                bufferOffset += (detectableObjects.Count + 2);
            }
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
