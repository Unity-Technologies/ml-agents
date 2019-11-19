using System;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents.Sensor
{
    public abstract class RayPerceptionSensorComponentBase : SensorComponent
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

        [Range(0f, 10f)]
        [Tooltip("Radius of sphere to cast. Set to zero for raycasts.")]
        public float sphereCastRadius = 0.5f;

        [Range(1, 1000)]
        [Tooltip("Length of the rays to cast.")]
        public float rayLength = 20f;

        [Range(1, 50)]
        [Tooltip("Whether to stack previous observations. Using 1 means no previous observations.")]
        public int observationStacks = 1;

        [Header("Debug Gizmos", order = 999)]
        public Color rayHitColor = Color.red;
        public Color rayMissColor = Color.white;
        [Tooltip("Whether to draw the raycasts in the world space of when they happened, or using the Agent's current transform'")]
        public bool useWorldPositions = true;


        [NonSerialized]
        RayPerceptionSensor m_RaySensor;

        public abstract RayPerceptionSensor.CastType GetCastType();

        public virtual float GetStartVerticalOffset()
        {
            return 0f;
        }

        public virtual float GetEndVerticalOffset()
        {
            return 0f;
        }

        public override ISensor CreateSensor()
        {
            var rayAngles = GetRayAngles(raysPerDirection, maxRayDegrees);
            m_RaySensor = new RayPerceptionSensor(sensorName, rayLength, detectableTags, rayAngles,
                transform, GetStartVerticalOffset(), GetEndVerticalOffset(), sphereCastRadius, GetCastType()
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
            var numTags = detectableTags == null ? 0 : detectableTags.Count;
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
                // Either use the original world-space coordinates of the raycast, or transform the agent-local
                // coordinates of the rays to the current transform of the agent. If the agent acts every frame,
                // these should be the same.
                var startPositionWorld = rayInfo.worldStart;
                var endPositionWorld = rayInfo.worldEnd;
                if (!useWorldPositions)
                {
                    startPositionWorld = transform.TransformPoint(rayInfo.localStart);
                    endPositionWorld = transform.TransformPoint(rayInfo.localEnd);
                }
                var rayDirection = endPositionWorld - startPositionWorld;
                rayDirection *= rayInfo.hitFraction;

                // hit fraction ^2 will shift "far" hits closer to the hit color
                var lerpT = rayInfo.hitFraction * rayInfo.hitFraction;
                var color = Color.Lerp(rayHitColor, rayMissColor, lerpT);
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
