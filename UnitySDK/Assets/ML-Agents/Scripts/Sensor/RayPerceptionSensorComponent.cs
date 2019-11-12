using System;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents.Sensor
{
    public class RayPerceptionSensorComponent : SensorComponent
    {
        public string sensorName = "RayPerceptionSensor";

        [Tooltip("List of tags in the scene to compare against.")]
        public List<string> detectableTags;

        [Range(0, 50)]
        [Tooltip("Number of rays to the left and right of center.")]
        public int raysPerDirection = 3;

        [Range(0, 180)]
        [Tooltip("Cone size for rays. Using 90 degrees will cast rays to the left and right. Greater than 90 degrees will go backwards.")]
        public float maxRayDegrees = 70;

        [Range(-10f, 10f)]
        [Tooltip("Ray start is offset up or down by this amount.")]
        public float startVerticalOffset;

        [Range(-10f, 10f)]
        [Tooltip("Ray end is offset up or down by this amount.")]
        public float endVerticalOffset;

        [Range(0f, 10f)]
        [Tooltip("Radius of sphere to cast. Set to zero for raycasts.")]
        public float sphereCastRadius = 0.5f;

        [Range(1, 1000)]
        [Tooltip("Length of the rays to cast.")]
        public float rayLength = 20f;

        [Range(1, 50)]
        [Tooltip("Whether to stack previous observations. Using 1 means no previous observations.")]
        public int observationStacks = 1;

        [NonSerialized]
        RayPerceptionSensor m_RaySensor;

        // TODO layerMask for raycasts

        public override ISensor CreateSensor()
        {
            var rayAngles = GetRayAngles(raysPerDirection, maxRayDegrees);
            m_RaySensor = new RayPerceptionSensor(sensorName, rayLength, detectableTags, rayAngles,
                transform, startVerticalOffset, endVerticalOffset, sphereCastRadius
            );

            if (observationStacks != 1)
            {
                var stackingSensor = new StackingSensor(m_RaySensor, observationStacks);
                return stackingSensor;
            }

            return m_RaySensor;
        }

        public static float[] GetRayAngles(int raysPerDirection, float maxRayDegrees)
        {
            // Example:
            // { 90, 90 - delta, 90 + delta, 90 - 2*delta, 90 + 2*delta }
            var anglesOut = new float[2 * raysPerDirection + 1];
            var delta = maxRayDegrees / raysPerDirection;
            anglesOut[0] = 90f;
            for (var i = 0; i < raysPerDirection; i++)
            {
                anglesOut[2 * i + 1] = 90 - (i+1) * delta;
                anglesOut[2 * i + 2] = 90 + (i+1) * delta;
            }
            return anglesOut;
        }

        public override int[] GetObservationShape()
        {
            var numRays = 2 * raysPerDirection + 1;
            var numTags = detectableTags.Count;
            var obsSize = (numTags + 2) * numRays;
            var stacks = observationStacks > 1 ? observationStacks : 1;
            return new[] { obsSize * stacks };
        }

        /// <summary>
        /// Draw the debug information from the sensor (if available).
        /// </summary>
        public void OnDrawGizmos()
        {
            if (m_RaySensor?.debugDisplayInfo?.rayInfos == null)
            {
                return;
            }
            var debugInfo = m_RaySensor.debugDisplayInfo;

            // Draw "old" observations in a lighter color.
            // Since the agent may not step every frame, this helps de-emphasize "stale" hit information.
            var alpha = Mathf.Pow(.5f, debugInfo.age);

            foreach (var rayInfo in debugInfo.rayInfos)
            {
                var startPositionWorld = transform.TransformPoint(rayInfo.localStart);
                var endPositionWorld = transform.TransformPoint(rayInfo.localEnd);
                var rayDirection = endPositionWorld - startPositionWorld;
                rayDirection *= rayInfo.hitFraction;

                var color = rayInfo.castHit ? Color.yellow : Color.blue;
                color.a = alpha;
                Gizmos.color = color;
                Gizmos.DrawRay(startPositionWorld,rayDirection);

                // Draw the hit point as a sphere. If using rays to cast (0 radius), use a small sphere.
                if (rayInfo.castHit)
                {
                    var hitRadius = Mathf.Max(sphereCastRadius, .05f);
                    Gizmos.DrawWireSphere(startPositionWorld + rayDirection, hitRadius);

                }
            }

        }
    }
}
