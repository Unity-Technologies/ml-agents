using System;
using System.Collections.Generic;
using UnityEngine;
using MLAgents.Sensor;

namespace MLAgents
{
    /// <summary>
    /// Ray 2D perception component. Attach this to agents to enable "local perception"
    /// via the use of ray casts directed outward from the agent.
    /// </summary>
    [Obsolete("The RayPerception MonoBehaviour is deprecated. Use the RayPerceptionSensorComponent instead")]
    public class RayPerception2D : RayPerception
    {
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
        /// <param name="startOffset">Unused</param>
        /// <param name="endOffset">Unused</param>
        public override IList<float> Perceive(float rayDistance,
            float[] rayAngles, string[] detectableObjects,
            float startOffset=0.0f, float endOffset=0.0f)
        {
            var perceptionSize = (detectableObjects.Length + 2) * rayAngles.Length;
            if (m_PerceptionBuffer == null || m_PerceptionBuffer.Length != perceptionSize)
            {
                m_PerceptionBuffer = new float[perceptionSize];
            }

            const float castRadius = 0.5f;
            const bool legacyHitFractionBehavior = true;
            RayPerceptionSensor.PerceiveStatic(
                rayDistance, rayAngles, detectableObjects, startOffset, endOffset, castRadius,
                transform, RayPerceptionSensor.CastType.Cast3D, m_PerceptionBuffer, legacyHitFractionBehavior
            );

            return m_PerceptionBuffer;
        }

    }
}
