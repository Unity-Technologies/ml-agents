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
        /// <summary>
        /// Length of the rays to cast. This will be scaled up or down based on the scale of the transform.
        /// </summary>
        public float rayLength;

        /// <summary>
        /// List of tags which correspond to object types agent can see.
        /// </summary>
        public IReadOnlyList<string> detectableObjects;

        /// <summary>
        /// List of angles (in degrees) used to define the rays.
        /// 90 degrees is considered "forward" relative to the game object.
        /// </summary>
        public IReadOnlyList<float> angles;

        /// <summary>
        /// Starting height offset of ray from center of agent
        /// </summary>
        public float startOffset;

        /// <summary>
        /// Ending height offset of ray from center of agent.
        /// </summary>
        public float endOffset;

        /// <summary>
        /// Radius of the sphere to use for spherecasting.
        /// If 0 or less, rays are used instead - this may be faster, especially for complex environments.
        /// </summary>
        public float castRadius;

        /// <summary>
        /// Transform of the GameObject.
        /// </summary>
        public Transform transform;

        /// <summary>
        /// Whether to perform the casts in 2D or 3D.
        /// </summary>
        public RayPerceptionCastType castType = RayPerceptionCastType.Cast3D;

        /// <summary>
        /// Filtering options for the casts.
        /// </summary>
        public int layerMask = Physics.DefaultRaycastLayers;

        /// <summary>
        /// Returns the expected number of floats in the output.
        /// </summary>
        /// <returns></returns>
        public int OutputSize()
        {
            return (detectableObjects.Count + 2) * angles.Count;
        }

        public (Vector3 StartPositionWorld, Vector3 EndPositionWorld) RayExtents(int rayIndex)
        {
            var angle = angles[rayIndex];
            Vector3 startPositionLocal, endPositionLocal;
            if (castType == RayPerceptionCastType.Cast3D)
            {
                startPositionLocal = new Vector3(0, startOffset, 0);
                endPositionLocal = PolarToCartesian3D(rayLength, angle);
                endPositionLocal.y += endOffset;
            }
            else
            {
                // Vector2s here get converted to Vector3s (and back to Vector2s for casting)
                startPositionLocal = new Vector2();
                endPositionLocal = PolarToCartesian2D(rayLength, angle);
            }

            var startPositionWorld = transform.TransformPoint(startPositionLocal);
            var endPositionWorld = transform.TransformPoint(endPositionLocal);

            return (StartPositionWorld: startPositionWorld, EndPositionWorld: endPositionWorld);
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

    public class RayPerceptionOutput
    {
        public struct RayOutput
        {
            /// <summary>
            /// Whether or not the ray hit anything.
            /// </summary>
            public bool hasHit;

            /// <summary>
            /// Whether or not the ray hit an object whose tag is in the input's detectableObjects list.
            /// </summary>
            public bool hitTaggedObject;

            /// <summary>
            /// The index of the hit object's tag in the detectableObjects list, or -1 if there was no hit, or the
            /// hit object has a different tag.
            /// </summary>
            public int hitTagIndex;

            /// <summary>
            /// Normalized distance to the hit object.
            /// </summary>
            public float hitFraction;

            /// <summary>
            /// World space location of the ray start. Only used for debugging.
            /// </summary>
            public Vector3 worldStart;

            /// <summary>
            /// World space location of the ray start. Only used for debugging
            /// </summary>
            public Vector3 worldEnd;

            /// <summary>
            /// The scaled castRadius that was actually used for casting.
            /// </summary>
            public float scaledCastRadius;

        }

        /// <summary>
        /// Writes the ray output information to a float array.  Each element in the rayAngles array determines a
        /// sublist of data to the observation. The sublist contains the observation data for a single cast.
        /// The list is composed of the following:
        /// 1. A one-hot encoding for detectable objects. For example, if detectableObjects.Length = n, the
        ///    first n elements of the sublist will be a one-hot encoding of the detectableObject that was hit, or
        ///    all zeroes otherwise.
        /// 2. The 'numDetectableObjects' element of the sublist will be 1 if the ray missed everything, or 0 if it hit
        ///    something (detectable or not).
        /// 3. The 'numDetectableObjects+1' element of the sublist will contain the normalized distance to the object
        ///    hit, or 1.0 if nothing was hit.
        /// </summary>
        /// <param name="numDetectableObjects"></param>
        /// <param name="buffer">Output buffer. The size must be equal to (numDetectableObjects+2) * rayOutputs.Length</param>
        public void ToFloatArray(int numDetectableObjects, float[] buffer)
        {
            Array.Clear(buffer, 0, buffer.Length);
            var bufferOffset = 0;

            for (var i = 0; i < rayOutputs.Length; i++)
            {
                var rayOutput = rayOutputs[i];
                if (rayOutput.hitTaggedObject)
                {
                    buffer[bufferOffset + rayOutput.hitTagIndex] = 1f;
                }
                buffer[bufferOffset + numDetectableObjects] = rayOutput.hasHit ? 0f : 1f;
                buffer[bufferOffset + numDetectableObjects + 1] = rayOutput.hitFraction;
                bufferOffset += numDetectableObjects + 2;
            }
        }

        /// <summary>
        /// RayOutput for each ray that was cast.
        /// </summary>
        public RayOutput[] rayOutputs;
    }

    public class RayPerceptionSensor : ISensor
    {
        float[] m_Observations;
        int[] m_Shape;
        string m_Name;

        RayPerceptionInput m_RayPerceptionInput;
        RayPerceptionOutput m_Output = new RayPerceptionOutput();

        /// <summary>
        /// Debug information for the raycast hits. This is used by the RayPerceptionSensorComponent.
        /// </summary>
        public class DebugDisplayInfo
        {
            public struct RayInfo
            {
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

        public RayPerceptionSensor(string name, RayPerceptionInput rayInput)
        {
            var numObservations = rayInput.OutputSize();
            m_Shape = new[] { numObservations };
            m_Name = name;
            m_RayPerceptionInput = rayInput;

            m_Observations = new float[numObservations];

            if (Application.isEditor)
            {
                m_DebugDisplayInfo = new DebugDisplayInfo();
            }
        }

        public int Write(WriteAdapter adapter)
        {
            using (TimerStack.Instance.Scoped("RayPerceptionSensor.Perceive"))
            {
                PerceiveStatic(m_RayPerceptionInput, m_Output, m_DebugDisplayInfo);
                m_Output.ToFloatArray(m_RayPerceptionInput.detectableObjects.Count, m_Observations);
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
        /// Evaluates the raycasts to be used as part of an observation of an agent.
        /// </summary>
        /// <param name="input">Input defining the rays that will be cast.</param>
        /// <param name="output">Output class that will be written to with raycast results.</param>
        /// <param name="debugInfo">Optional debug information output, only used by RayPerceptionSensor.</param>
        ///
        public static void PerceiveStatic(RayPerceptionInput input,
            RayPerceptionOutput output,
            DebugDisplayInfo debugInfo = null)
        {

            if (output.rayOutputs == null || output.rayOutputs.Length != input.angles.Count)
            {
                output.rayOutputs = new RayPerceptionOutput.RayOutput[input.angles.Count];
            }

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
            for (var rayIndex = 0; rayIndex < input.angles.Count; rayIndex++)
            {
                var extents = input.RayExtents(rayIndex);
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
                bool castHit;
                float hitFraction;
                GameObject hitObject;
                var rayOutput = new RayPerceptionOutput.RayOutput();

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

                rayOutput.hasHit = castHit;
                rayOutput.hitFraction = hitFraction;
                rayOutput.hitTaggedObject = false;
                rayOutput.hitTagIndex = -1;

                if (castHit)
                {
                    for (var i = 0; i < input.detectableObjects.Count; i++)
                    {
                        if (hitObject.CompareTag(input.detectableObjects[i]))
                        {
                            rayOutput.hitTaggedObject = true;
                            rayOutput.hitTagIndex = i;
                            break;
                        }
                    }
                }

                output.rayOutputs[rayIndex] = rayOutput;
            }
        }
    }
}
