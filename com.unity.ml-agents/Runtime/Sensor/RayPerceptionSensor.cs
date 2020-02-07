using System;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    public class RayPerceptionSensor : ISensor
    {
        public enum CastType
        {
            Cast2D,
            Cast3D,
        }

        float[] m_Observations;
        int[] m_Shape;
        string m_Name;

        float m_RayDistance;
        List<string> m_DetectableObjects;
        float[] m_Angles;

        float m_StartOffset;
        float m_EndOffset;
        float m_CastRadius;
        CastType m_CastType;
        Transform m_Transform;
        int m_LayerMask;

        /// <summary>
        /// Debug information for the raycast hits. This is used by the RayPerceptionSensorComponent.
        /// </summary>
        public class DebugDisplayInfo
        {
            public struct RayInfo
            {
                public Vector3 localStart;
                public Vector3 localEnd;
                public Vector3 worldStart;
                public Vector3 worldEnd;
                public bool castHit;
                public float hitFraction;
                public float castRadius;
            }

            public void Reset()
            {
                m_Frame = Time.frameCount;
            }

            /// <summary>
            /// "Age" of the results in number of frames. This is used to adjust the alpha when drawing.
            /// </summary>
            public int age
            {
                get { return Time.frameCount - m_Frame; }
            }

            public RayInfo[] rayInfos;

            int m_Frame;
        }

        DebugDisplayInfo m_DebugDisplayInfo;

        public DebugDisplayInfo debugDisplayInfo
        {
            get { return m_DebugDisplayInfo; }
        }

        public RayPerceptionSensor(string name, float rayDistance, List<string> detectableObjects, float[] angles,
                                   Transform transform, float startOffset, float endOffset, float castRadius, CastType castType,
                                   int rayLayerMask)
        {
            var numObservations = (detectableObjects.Count + 2) * angles.Length;
            m_Shape = new[] { numObservations };
            m_Name = name;

            m_Observations = new float[numObservations];

            m_RayDistance = rayDistance;
            m_DetectableObjects = detectableObjects;
            // TODO - preprocess angles, save ray directions instead?
            m_Angles = angles;
            m_Transform = transform;
            m_StartOffset = startOffset;
            m_EndOffset = endOffset;
            m_CastRadius = castRadius;
            m_CastType = castType;
            m_LayerMask = rayLayerMask;

            if (Application.isEditor)
            {
                m_DebugDisplayInfo = new DebugDisplayInfo();
            }
        }

        public int Write(WriteAdapter adapter)
        {
            using (TimerStack.Instance.Scoped("RayPerceptionSensor.Perceive"))
            {
                PerceiveStatic(
                    m_RayDistance, m_Angles, m_DetectableObjects, m_StartOffset, m_EndOffset,
                    m_CastRadius, m_Transform, m_CastType, m_Observations, m_LayerMask,
                    m_DebugDisplayInfo
                );
                adapter.AddRange(m_Observations);
            }
            return m_Observations.Length;
        }

        public void Update()
        {
        }

        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        public string GetName()
        {
            return m_Name;
        }

        public virtual byte[] GetCompressedObservation()
        {
            return null;
        }

        public virtual SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.None;
        }

        /// <summary>
        /// Evaluates a perception vector to be used as part of an observation of an agent.
        /// Each element in the rayAngles array determines a sublist of data to the observation.
        /// The sublist contains the observation data for a single cast. The list is composed of the following:
        /// 1. A one-hot encoding for detectable objects. For example, if detectableObjects.Length = n, the
        ///    first n elements of the sublist will be a one-hot encoding of the detectableObject that was hit, or
        ///    all zeroes otherwise.
        /// 2. The 'length' element of the sublist will be 1 if the ray missed everything, or 0 if it hit
        ///    something (detectable or not).
        /// 3. The 'length+1' element of the sublist will contain the normalised distance to the object hit, or 1 if
        ///    nothing was hit.
        ///
        /// </summary>
        /// <param name="unscaledRayLength"></param>
        /// <param name="rayAngles">List of angles (in degrees) used to define the rays. 90 degrees is considered
        ///     "forward" relative to the game object</param>
        /// <param name="detectableObjects">List of tags which correspond to object types agent can see</param>
        /// <param name="startOffset">Starting height offset of ray from center of agent.</param>
        /// <param name="endOffset">Ending height offset of ray from center of agent.</param>
        /// <param name="unscaledCastRadius">Radius of the sphere to use for spherecasting. If 0 or less, rays are used
        /// instead - this may be faster, especially for complex environments.</param>
        /// <param name="transform">Transform of the GameObject</param>
        /// <param name="castType">Whether to perform the casts in 2D or 3D.</param>
        /// <param name="perceptionBuffer">Output array of floats. Must be (num rays) * (num tags + 2) in size.</param>
        /// <param name="layerMask">Filtering options for the casts</param>
        /// <param name="debugInfo">Optional debug information output, only used by RayPerceptionSensor.</param>
        ///
        public static void PerceiveStatic(float unscaledRayLength,
            IReadOnlyList<float> rayAngles, IReadOnlyList<string> detectableObjects,
            float startOffset, float endOffset, float unscaledCastRadius,
            Transform transform, CastType castType, float[] perceptionBuffer,
            int layerMask = Physics.DefaultRaycastLayers,
            DebugDisplayInfo debugInfo = null)
        {
            Array.Clear(perceptionBuffer, 0, perceptionBuffer.Length);
            if (debugInfo != null)
            {
                debugInfo.Reset();
                if (debugInfo.rayInfos == null || debugInfo.rayInfos.Length != rayAngles.Count)
                {
                    debugInfo.rayInfos = new DebugDisplayInfo.RayInfo[rayAngles.Count];
                }
            }

            // For each ray sublist stores categorical information on detected object
            // along with object distance.
            int bufferOffset = 0;
            for (var rayIndex = 0; rayIndex < rayAngles.Count; rayIndex++)
            {
                var angle = rayAngles[rayIndex];
                Vector3 startPositionLocal, endPositionLocal;
                if (castType == CastType.Cast3D)
                {
                    startPositionLocal = new Vector3(0, startOffset, 0);
                    endPositionLocal = PolarToCartesian3D(unscaledRayLength, angle);
                    endPositionLocal.y += endOffset;
                }
                else
                {
                    // Vector2s here get converted to Vector3s (and back to Vector2s for casting)
                    startPositionLocal = new Vector2();
                    endPositionLocal = PolarToCartesian2D(unscaledRayLength, angle);
                }

                var startPositionWorld = transform.TransformPoint(startPositionLocal);
                var endPositionWorld = transform.TransformPoint(endPositionLocal);

                var rayDirection = endPositionWorld - startPositionWorld;
                // If there is non-unity scale, |rayDirection| will be different from rayLength.
                // We want to use this transformed ray length for determining cast length, hit fraction etc.
                // We also it to scale up or down the sphere or circle radii
                var scaledRayLength = rayDirection.magnitude;
                // Avoid 0/0 if unscaledRayLength is 0
                var scaledCastRadius = unscaledRayLength > 0 ? unscaledCastRadius * scaledRayLength / unscaledRayLength : unscaledCastRadius;

                // Do the cast and assign the hit information for each detectable object.
                //     sublist[0           ] <- did hit detectableObjects[0]
                //     ...
                //     sublist[numObjects-1] <- did hit detectableObjects[numObjects-1]
                //     sublist[numObjects  ] <- 1 if missed else 0
                //     sublist[numObjects+1] <- hit fraction (or 1 if no hit)

                bool castHit;
                float hitFraction;
                GameObject hitObject;

                if (castType == CastType.Cast3D)
                {
                    RaycastHit rayHit;
                    if (scaledCastRadius > 0f)
                    {
                        castHit = Physics.SphereCast(startPositionWorld, scaledCastRadius, rayDirection, out rayHit,
                            scaledRayLength, layerMask);
                    }
                    else
                    {
                        castHit = Physics.Raycast(startPositionWorld, rayDirection, out rayHit,
                            scaledRayLength, layerMask);
                    }

                    // If scaledRayLength is 0, we still could have a hit with sphere casts (maybe?).
                    // To avoid 0/0, set the fraction to 0.
                    hitFraction = castHit ? (scaledRayLength > 0 ? rayHit.distance / scaledRayLength : 0.0f) : 1.0f;
                    hitObject = castHit ? rayHit.collider.gameObject : null;
                }
                else
                {
                    RaycastHit2D rayHit;
                    if (scaledCastRadius > 0f)
                    {
                        rayHit = Physics2D.CircleCast(startPositionWorld, scaledCastRadius, rayDirection,
                            scaledRayLength, layerMask);
                    }
                    else
                    {
                        rayHit = Physics2D.Raycast(startPositionWorld, rayDirection, scaledRayLength, layerMask);
                    }

                    castHit = rayHit;
                    hitFraction = castHit ? rayHit.fraction : 1.0f;
                    hitObject = castHit ? rayHit.collider.gameObject : null;
                }

                if (debugInfo != null)
                {
                    debugInfo.rayInfos[rayIndex].localStart = startPositionLocal;
                    debugInfo.rayInfos[rayIndex].localEnd = endPositionLocal;
                    debugInfo.rayInfos[rayIndex].worldStart = startPositionWorld;
                    debugInfo.rayInfos[rayIndex].worldEnd = endPositionWorld;
                    debugInfo.rayInfos[rayIndex].castHit = castHit;
                    debugInfo.rayInfos[rayIndex].hitFraction = hitFraction;
                    debugInfo.rayInfos[rayIndex].castRadius = scaledCastRadius;
                }
                else if (Application.isEditor)
                {
                    // Legacy drawing
                    Debug.DrawRay(startPositionWorld, rayDirection, Color.black, 0.01f, true);
                }

                if (castHit)
                {
                    bool hitTaggedObject = false;
                    for (var i = 0; i < detectableObjects.Count; i++)
                    {
                        if (hitObject.CompareTag(detectableObjects[i]))
                        {
                            perceptionBuffer[bufferOffset + i] = 1;
                            perceptionBuffer[bufferOffset + detectableObjects.Count + 1] = hitFraction;
                            hitTaggedObject = true;
                            break;
                        }
                    }

                    if (!hitTaggedObject)
                    {
                        // Something was hit but not on the list. Still set the hit fraction.
                        perceptionBuffer[bufferOffset + detectableObjects.Count + 1] = hitFraction;
                    }
                }
                else
                {
                    perceptionBuffer[bufferOffset + detectableObjects.Count] = 1f;
                    // Nothing was hit, so there's full clearance in front of the agent.
                    perceptionBuffer[bufferOffset + detectableObjects.Count + 1] = 1.0f;
                }

                bufferOffset += detectableObjects.Count + 2;
            }
        }

        /// <summary>
        /// Converts polar coordinate to cartesian coordinate.
        /// </summary>
        static Vector3 PolarToCartesian3D(float radius, float angleDegrees)
        {
            var x = radius * Mathf.Cos(Mathf.Deg2Rad * angleDegrees);
            var z = radius * Mathf.Sin(Mathf.Deg2Rad * angleDegrees);
            return new Vector3(x, 0f, z);
        }

        /// <summary>
        /// Converts polar coordinate to cartesian coordinate.
        /// </summary>
        static Vector2 PolarToCartesian2D(float radius, float angleDegrees)
        {
            var x = radius * Mathf.Cos(Mathf.Deg2Rad * angleDegrees);
            var y = radius * Mathf.Sin(Mathf.Deg2Rad * angleDegrees);
            return new Vector2(x, y);
        }
    }
}
