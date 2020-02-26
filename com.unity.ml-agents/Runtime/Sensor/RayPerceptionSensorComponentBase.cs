using System;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// A base class to support sensor components for raycast-based sensors.
    /// </summary>
    public abstract class RayPerceptionSensorComponentBase : SensorComponent
    {
        /// <summary>
        /// The name of the Sensor that this component wraps.
        /// </summary>
        public string sensorName = "RayPerceptionSensor";

        /// <summary>
        /// List of tags in the scene to compare against.
        /// </summary>
        [Tooltip("List of tags in the scene to compare against.")]
        public List<string> detectableTags;

        /// <summary>
        /// Number of rays to the left and right of center.
        /// </summary>
        [Range(0, 50)]
        [Tooltip("Number of rays to the left and right of center.")]
        public int raysPerDirection = 3;

        /// <summary>
        /// Cone size for rays. Using 90 degrees will cast rays to the left and right.
        /// Greater than 90 degrees will go backwards.
        /// </summary>
        [Range(0, 180)]
        [Tooltip("Cone size for rays. Using 90 degrees will cast rays to the left and right. " +
                 "Greater than 90 degrees will go backwards.")]
        public float maxRayDegrees = 70;

        /// <summary>
        /// Radius of sphere to cast. Set to zero for raycasts.
        /// </summary>
        [Range(0f, 10f)]
        [Tooltip("Radius of sphere to cast. Set to zero for raycasts.")]
        public float sphereCastRadius = 0.5f;

        /// <summary>
        /// Length of the rays to cast.
        /// </summary>
        [Range(1, 1000)]
        [Tooltip("Length of the rays to cast.")]
        public float rayLength = 20f;

        /// <summary>
        /// Controls which layers the rays can hit.
        /// </summary>
        [Tooltip("Controls which layers the rays can hit.")]
        public LayerMask rayLayerMask = Physics.DefaultRaycastLayers;

        /// <summary>
        /// Whether to stack previous observations. Using 1 means no previous observations.
        /// </summary>
        [Range(1, 50)]
        [Tooltip("Whether to stack previous observations. Using 1 means no previous observations.")]
        public int observationStacks = 1;

        /// <summary>
        /// Color to code a ray that hits another object.
        /// </summary>
        [Header("Debug Gizmos", order = 999)]
        public Color rayHitColor = Color.red;

        /// <summary>
        /// Color to code a ray that avoid or misses all other objects.
        /// </summary>
        public Color rayMissColor = Color.white;

        [NonSerialized]
        RayPerceptionSensor m_RaySensor;

        /// <summary>
        /// Returns the <see cref="RayPerceptionCastType"/> for the associated raycast sensor.
        /// </summary>
        /// <returns></returns>
        public abstract RayPerceptionCastType GetCastType();

        /// <summary>
        /// Returns the amount that the ray start is offset up or down by.
        /// </summary>
        /// <returns></returns>
        public virtual float GetStartVerticalOffset()
        {
            return 0f;
        }

        /// <summary>
        /// Returns the amount that the ray end is offset up or down by.
        /// </summary>
        /// <returns></returns>
        public virtual float GetEndVerticalOffset()
        {
            return 0f;
        }

        /// <summary>
        /// Returns an initialized raycast sensor.
        /// </summary>
        /// <returns></returns>
        public override ISensor CreateSensor()
        {
            var rayAngles = GetRayAngles(raysPerDirection, maxRayDegrees);

            var rayPerceptionInput = new RayPerceptionInput
            {
                rayLength = rayLength,
                detectableTags = detectableTags,
                angles = rayAngles,
                startOffset = GetStartVerticalOffset(),
                endOffset = GetEndVerticalOffset(),
                castRadius = sphereCastRadius,
                transform = transform,
                castType = GetCastType(),
                layerMask = rayLayerMask
            };

            m_RaySensor = new RayPerceptionSensor(sensorName, rayPerceptionInput);

            if (observationStacks != 1)
            {
                var stackingSensor = new StackingSensor(m_RaySensor, observationStacks);
                return stackingSensor;
            }

            return m_RaySensor;
        }

        /// <summary>
        /// Returns the specific ray angles given the number of rays per direction and the
        /// cone size for the rays.
        /// </summary>
        /// <param name="raysPerDirection">Number of rays to the left and right of center.</param>
        /// <param name="maxRayDegrees">
        /// Cone size for rays. Using 90 degrees will cast rays to the left and right.
        /// Greater than 90 degrees will go backwards.
        /// </param>
        /// <returns></returns>
        public static float[] GetRayAngles(int raysPerDirection, float maxRayDegrees)
        {
            // Example:
            // { 90, 90 - delta, 90 + delta, 90 - 2*delta, 90 + 2*delta }
            var anglesOut = new float[2 * raysPerDirection + 1];
            var delta = maxRayDegrees / raysPerDirection;
            anglesOut[0] = 90f;
            for (var i = 0; i < raysPerDirection; i++)
            {
                anglesOut[2 * i + 1] = 90 - (i + 1) * delta;
                anglesOut[2 * i + 2] = 90 + (i + 1) * delta;
            }
            return anglesOut;
        }

        /// <summary>
        /// Returns the observation shape for this raycast sensor which depends on the number
        /// of tags for detected objects and the number of rays.
        /// </summary>
        /// <returns></returns>
        public override int[] GetObservationShape()
        {
            var numRays = 2 * raysPerDirection + 1;
            var numTags = detectableTags?.Count ?? 0;
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
                var startPositionWorld = rayInfo.worldStart;
                var endPositionWorld = rayInfo.worldEnd;
                var rayDirection = endPositionWorld - startPositionWorld;
                rayDirection *= rayInfo.rayOutput.hitFraction;

                // hit fraction ^2 will shift "far" hits closer to the hit color
                var lerpT = rayInfo.rayOutput.hitFraction * rayInfo.rayOutput.hitFraction;
                var color = Color.Lerp(rayHitColor, rayMissColor, lerpT);
                color.a *= alpha;
                Gizmos.color = color;
                Gizmos.DrawRay(startPositionWorld, rayDirection);

                // Draw the hit point as a sphere. If using rays to cast (0 radius), use a small sphere.
                if (rayInfo.rayOutput.hasHit)
                {
                    var hitRadius = Mathf.Max(rayInfo.castRadius, .05f);
                    Gizmos.DrawWireSphere(startPositionWorld + rayDirection, hitRadius);
                }
            }
        }
    }
}
