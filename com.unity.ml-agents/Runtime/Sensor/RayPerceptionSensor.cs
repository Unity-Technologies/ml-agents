using System;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents
{
    public enum RayPerceptionCastType
    {
        Cast2D,
        Cast3D,
    }

    public class RayPerceptionInput
    {
        // TODO
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

        public float rayLength;
        public IReadOnlyList<string> detectableObjects;
        public IReadOnlyList<float> angles;

        public float startOffset;
        public float endOffset;
        public float castRadius;
        public RayPerceptionCastType castType = RayPerceptionCastType.Cast3D;

        public int layerMask = Physics.DefaultRaycastLayers;

        public int OutputSize()
        {
            return (detectableObjects.Count + 2) * angles.Count;
        }

        public (Vector3 StartPositionWorld, Vector3 EndPositionWorld) RayExtents(int rayIndex, Transform transform)
        {
            var angle = angles[rayIndex];
            Vector3 startPositionLocal, endPositionLocal;
            if (castType == RayPerceptionCastType.Cast3D)
            {
                startPositionLocal = new Vector3(0, startOffset, 0);
                endPositionLocal = RayPerceptionSensor.PolarToCartesian3D(rayLength, angle);
                endPositionLocal.y += endOffset;
            }
            else
            {
                // Vector2s here get converted to Vector3s (and back to Vector2s for casting)
                startPositionLocal = new Vector2();
                endPositionLocal = RayPerceptionSensor.PolarToCartesian2D(rayLength, angle);
            }

            var startPositionWorld = transform.TransformPoint(startPositionLocal);
            var endPositionWorld = transform.TransformPoint(endPositionLocal);

            return (StartPositionWorld: startPositionWorld, EndPositionWorld: endPositionWorld);
        }

    }

    public class RayPerceptionSensor : ISensor
    {
        float[] m_Observations;
        int[] m_Shape;
        string m_Name;

        RayPerceptionInput m_RayPerceptionInput;
        Transform m_Transform;

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

        public RayPerceptionSensor(string name, Transform transform, RayPerceptionInput rayInput)
        {
            var numObservations = rayInput.OutputSize();
            m_Shape = new[] { numObservations };
            m_Name = name;
            m_RayPerceptionInput = rayInput;

            m_Observations = new float[numObservations];

            m_Transform = transform;

            if (Application.isEditor)
            {
                m_DebugDisplayInfo = new DebugDisplayInfo();
            }
        }

        public int Write(WriteAdapter adapter)
        {
            using (TimerStack.Instance.Scoped("RayPerceptionSensor.Perceive"))
            {
                PerceiveStatic(m_RayPerceptionInput, m_Transform, m_Observations, m_DebugDisplayInfo);
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

        ///
        public static void PerceiveStatic(RayPerceptionInput input,
            Transform transform, float[] perceptionBuffer,
            DebugDisplayInfo debugInfo = null)
        {
            Array.Clear(perceptionBuffer, 0, perceptionBuffer.Length);
            if (debugInfo != null)
            {
                debugInfo.Reset();
                if (debugInfo.rayInfos == null || debugInfo.rayInfos.Length != input.angles.Count)
                {
                    debugInfo.rayInfos = new DebugDisplayInfo.RayInfo[input.angles.Count];
                }
            }

            // For each ray sublist stores categorical information on detected object
            // along with object distance.
            var unscaledRayLength = input.rayLength;
            var unscaledCastRadius = input.castRadius;
            int bufferOffset = 0;
            for (var rayIndex = 0; rayIndex < input.angles.Count; rayIndex++)
            {
                var extents = input.RayExtents(rayIndex, transform);
                var startPositionWorld = extents.StartPositionWorld;
                var endPositionWorld = extents.EndPositionWorld;

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

                if (input.castType == RayPerceptionCastType.Cast3D)
                {
                    RaycastHit rayHit;
                    if (scaledCastRadius > 0f)
                    {
                        castHit = Physics.SphereCast(startPositionWorld, scaledCastRadius, rayDirection, out rayHit,
                            scaledRayLength, input.layerMask);
                    }
                    else
                    {
                        castHit = Physics.Raycast(startPositionWorld, rayDirection, out rayHit,
                            scaledRayLength, input.layerMask);
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
                            scaledRayLength, input.layerMask);
                    }
                    else
                    {
                        rayHit = Physics2D.Raycast(startPositionWorld, rayDirection, scaledRayLength, input.layerMask);
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
                    for (var i = 0; i <input.detectableObjects.Count; i++)
                    {
                        if (hitObject.CompareTag(input.detectableObjects[i]))
                        {
                            perceptionBuffer[bufferOffset + i] = 1;
                            perceptionBuffer[bufferOffset + input.detectableObjects.Count + 1] = hitFraction;
                            hitTaggedObject = true;
                            break;
                        }
                    }

                    if (!hitTaggedObject)
                    {
                        // Something was hit but not on the list. Still set the hit fraction.
                        perceptionBuffer[bufferOffset + input.detectableObjects.Count + 1] = hitFraction;
                    }
                }
                else
                {
                    perceptionBuffer[bufferOffset + input.detectableObjects.Count] = 1f;
                    // Nothing was hit, so there's full clearance in front of the agent.
                    perceptionBuffer[bufferOffset + input.detectableObjects.Count + 1] = 1.0f;
                }

                bufferOffset += input.detectableObjects.Count + 2;
            }
        }

        /// <summary>
        /// Converts polar coordinate to cartesian coordinate.
        /// </summary>
        static internal Vector3 PolarToCartesian3D(float radius, float angleDegrees)
        {
            var x = radius * Mathf.Cos(Mathf.Deg2Rad * angleDegrees);
            var z = radius * Mathf.Sin(Mathf.Deg2Rad * angleDegrees);
            return new Vector3(x, 0f, z);
        }

        /// <summary>
        /// Converts polar coordinate to cartesian coordinate.
        /// </summary>
        static internal Vector2 PolarToCartesian2D(float radius, float angleDegrees)
        {
            var x = radius * Mathf.Cos(Mathf.Deg2Rad * angleDegrees);
            var y = radius * Mathf.Sin(Mathf.Deg2Rad * angleDegrees);
            return new Vector2(x, y);
        }
    }
}
