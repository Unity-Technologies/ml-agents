using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;

namespace MLAgents
{
    public abstract class RayPerceptionSensorComponentBase : SensorComponent
    {
        [HideInInspector]
        [SerializeField]
        [FormerlySerializedAs("sensorName")]
        string m_SensorName = "RayPerceptionSensor";
        internal string sensorName
        {
            get => m_SensorName;
            set => m_SensorName = value;
        }

        //[HideInInspector]
        [SerializeField]
        [FormerlySerializedAs("detectableTags")]
        [Tooltip("List of tags in the scene to compare against.")]
        List<string> m_DetectableTags;
        public List<string> detectableTags
        {
            get => m_DetectableTags;
            set => m_DetectableTags = value; // Note: can't change at runtime
        }

        [HideInInspector]
        [SerializeField]
        [FormerlySerializedAs("raysPerDirection")]
        [Range(0, 50)]
        [Tooltip("Number of rays to the left and right of center.")]
        int m_RaysPerDirection = 3;
        internal int raysPerDirection
        {
            get => m_RaysPerDirection;
            set => m_RaysPerDirection = value; // Note: can't change at runtime
        }

        [HideInInspector]
        [SerializeField]
        [FormerlySerializedAs("maxRayDegrees")]
        [Range(0, 180)]
        [Tooltip("Cone size for rays. Using 90 degrees will cast rays to the left and right. Greater than 90 degrees will go backwards.")]
        float m_MaxRayDegrees = 70;
        public float maxRayDegrees
        {
            get => m_MaxRayDegrees;
            set { m_MaxRayDegrees = value; UpdateSensor(); }
        }

        [HideInInspector]
        [SerializeField]
        [FormerlySerializedAs("sphereCastRadius")]
        [Range(0f, 10f)]
        [Tooltip("Radius of sphere to cast. Set to zero for raycasts.")]
        float m_SphereCastRadius = 0.5f;
        public float sphereCastRadius
        {
            get => m_SphereCastRadius;
            set { m_SphereCastRadius = value; UpdateSensor(); }
        }

        [HideInInspector]
        [SerializeField]
        [FormerlySerializedAs("rayLength")]
        [Range(1, 1000)]
        [Tooltip("Length of the rays to cast.")]
        float m_RayLength = 20f;
        public float rayLength
        {
            get => m_RayLength;
            set { m_RayLength = value; UpdateSensor(); }
        }

        [HideInInspector]
        [SerializeField]
        [FormerlySerializedAs("rayLayerMask")]
        [Tooltip("Controls which layers the rays can hit.")]
        LayerMask m_RayLayerMask = Physics.DefaultRaycastLayers;
        public LayerMask rayLayerMask
        {
            get => m_RayLayerMask;
            set { m_RayLayerMask = value; UpdateSensor();}
        }

        [HideInInspector]
        [SerializeField]
        [FormerlySerializedAs("observationStacks")]
        [Range(1, 50)]
        [Tooltip("Whether to stack previous observations. Using 1 means no previous observations.")]
        int m_ObservationStacks = 1;
        internal int observationStacks
        {
            get => m_ObservationStacks;
            set => m_ObservationStacks = value; // Note: can't change at runtime
        }

        [HideInInspector]
        [SerializeField]
        [Header("Debug Gizmos", order = 999)]
        internal Color rayHitColor = Color.red;

        [HideInInspector]
        [SerializeField]
        internal Color rayMissColor = Color.white;

        [NonSerialized]
        RayPerceptionSensor m_RaySensor;

        public abstract RayPerceptionCastType GetCastType();

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
            var rayPerceptionInput = GetRayPerceptionInput();

            m_RaySensor = new RayPerceptionSensor(m_SensorName, rayPerceptionInput);

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
                anglesOut[2 * i + 1] = 90 - (i + 1) * delta;
                anglesOut[2 * i + 2] = 90 + (i + 1) * delta;
            }
            return anglesOut;
        }

        public override int[] GetObservationShape()
        {
            var numRays = 2 * raysPerDirection + 1;
            var numTags = m_DetectableTags?.Count ?? 0;
            var obsSize = (numTags + 2) * numRays;
            var stacks = observationStacks > 1 ? observationStacks : 1;
            return new[] { obsSize * stacks };
        }

        RayPerceptionInput GetRayPerceptionInput()
        {
            var rayAngles = GetRayAngles(raysPerDirection, maxRayDegrees);

            var rayPerceptionInput = new RayPerceptionInput();
            rayPerceptionInput.rayLength = rayLength;
            rayPerceptionInput.detectableTags = detectableTags;
            rayPerceptionInput.angles = rayAngles;
            rayPerceptionInput.startOffset = GetStartVerticalOffset();
            rayPerceptionInput.endOffset = GetEndVerticalOffset();
            rayPerceptionInput.castRadius = sphereCastRadius;
            rayPerceptionInput.transform = transform;
            rayPerceptionInput.castType = GetCastType();
            rayPerceptionInput.layerMask = rayLayerMask;

            return rayPerceptionInput;
        }

        internal void UpdateSensor()
        {
            if (m_RaySensor != null)
            {
                var rayInput = GetRayPerceptionInput();
                m_RaySensor.SetRayPerceptionInput(rayInput);
            }
        }

        void OnDrawGizmosSelected()
        {
            if (m_RaySensor?.debugDisplayInfo?.rayInfos != null)
            {
                // If we have cached debug info from the sensor, draw that.
                // Draw "old" observations in a lighter color.
                // Since the agent may not step every frame, this helps de-emphasize "stale" hit information.
                var alpha = Mathf.Pow(.5f, m_RaySensor.debugDisplayInfo.age);

                foreach (var rayInfo in m_RaySensor.debugDisplayInfo.rayInfos)
                {
                    DrawRaycastGizmos(rayInfo, alpha);
                }
            }
            else
            {
                var rayInput = GetRayPerceptionInput();
                for (var rayIndex = 0; rayIndex < rayInput.angles.Count; rayIndex++)
                {
                    DebugDisplayInfo.RayInfo debugRay;
                    RayPerceptionSensor.PerceiveSingleRay(rayInput, rayIndex, out debugRay);
                    DrawRaycastGizmos(debugRay);
                }
            }
        }

        /// <summary>
        /// Draw the debug information from the sensor (if available).
        /// </summary>
        void DrawRaycastGizmos(DebugDisplayInfo.RayInfo rayInfo, float alpha=1.0f)
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
