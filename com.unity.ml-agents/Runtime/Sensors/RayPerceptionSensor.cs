using System;
using System.Collections.Generic;
using UnityEngine;

namespace MLAgents.Sensors
{
    /// <summary>
    /// Determines which dimensions the sensor will perform the casts in.
    /// </summary>
    public enum RayPerceptionCastType
    {
        /// <summary>
        /// Cast in 2 dimensions, using Physics2D.CircleCast or Physics2D.RayCast.
        /// </summary>
        Cast2D,

        /// <summary>
        /// Cast in 3 dimensions, using Physics.SphereCast or Physics.RayCast.
        /// </summary>
        Cast3D,
    }

    /// <summary>
    /// Contains the elements that define a ray perception sensor.
    /// </summary>
    public struct RayPerceptionInput
    {
        /// <summary>
        /// Length of the rays to cast. This will be scaled up or down based on the scale of the transform.
        /// </summary>
        public float rayLength;

        /// <summary>
        /// List of tags which correspond to object types agent can see.
        /// </summary>
        public IReadOnlyList<string> detectableTags;

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
        public RayPerceptionCastType castType;

        /// <summary>
        /// Filtering options for the casts.
        /// </summary>
        public int layerMask;

        /// <summary>
        /// Returns the expected number of floats in the output.
        /// </summary>
        /// <returns></returns>
        public int OutputSize()
        {
            return (detectableTags.Count + 2) * angles.Count;
        }

        /// <summary>
        /// Get the cast start and end points for the given ray index/
        /// </summary>
        /// <param name="rayIndex"></param>
        /// <returns>A tuple of the start and end positions in world space.</returns>
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

            return (StartPositionWorld : startPositionWorld, EndPositionWorld : endPositionWorld);
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

    /// <summary>
    /// Contains the data generated/produced from a ray perception sensor.
    /// </summary>
    public class RayPerceptionOutput
    {
        /// <summary>
        /// Contains the data generated from a single ray of a ray perception sensor.
        /// </summary>
        public struct RayOutput
        {
            /// <summary>
            /// Whether or not the ray hit anything.
            /// </summary>
            public bool hasHit;

            /// <summary>
            /// Whether or not the ray hit an object whose tag is in the input's detectableTags list.
            /// </summary>
            public bool hitTaggedObject;

            /// <summary>
            /// The index of the hit object's tag in the detectableTags list, or -1 if there was no hit, or the
            /// hit object has a different tag.
            /// </summary>
            public int hitTagIndex;

            /// <summary>
            /// Normalized distance to the hit object.
            /// </summary>
            public float hitFraction;

            /// <summary>
            /// Writes the ray output information to a subset of the float array.  Each element in the rayAngles array
            /// determines a sublist of data to the observation. The sublist contains the observation data for a single cast.
            /// The list is composed of the following:
            /// 1. A one-hot encoding for detectable tags. For example, if detectableTags.Length = n, the
            ///    first n elements of the sublist will be a one-hot encoding of the detectableTag that was hit, or
            ///    all zeroes otherwise.
            /// 2. The 'numDetectableTags' element of the sublist will be 1 if the ray missed everything, or 0 if it hit
            ///    something (detectable or not).
            /// 3. The 'numDetectableTags+1' element of the sublist will contain the normalized distance to the object
            ///    hit, or 1.0 if nothing was hit.
            /// </summary>
            /// <param name="numDetectableTags"></param>
            /// <param name="rayIndex"></param>
            /// <param name="buffer">Output buffer. The size must be equal to (numDetectableTags+2) * rayOutputs.Length</param>
            public void ToFloatArray(int numDetectableTags, int rayIndex, float[] buffer)
            {
                var bufferOffset = (numDetectableTags + 2) * rayIndex;
                if (hitTaggedObject)
                {
                    buffer[bufferOffset + hitTagIndex] = 1f;
                }
                buffer[bufferOffset + numDetectableTags] = hasHit ? 0f : 1f;
                buffer[bufferOffset + numDetectableTags + 1] = hitFraction;
            }
        }

        /// <summary>
        /// RayOutput for each ray that was cast.
        /// </summary>
        public RayOutput[] rayOutputs;
    }

    /// <summary>
    /// Debug information for the raycast hits. This is used by the RayPerceptionSensorComponent.
    /// </summary>
    internal class DebugDisplayInfo
    {
        public struct RayInfo
        {
            public Vector3 worldStart;
            public Vector3 worldEnd;
            public float castRadius;
            public RayPerceptionOutput.RayOutput rayOutput;
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

    /// <summary>
    /// A sensor implementation that supports ray cast-based observations.
    /// </summary>
    public class RayPerceptionSensor : ISensor
    {
        float[] m_Observations;
        int[] m_Shape;
        string m_Name;

        RayPerceptionInput m_RayPerceptionInput;

        DebugDisplayInfo m_DebugDisplayInfo;

        internal DebugDisplayInfo debugDisplayInfo
        {
            get { return m_DebugDisplayInfo; }
        }

        /// <summary>
        /// Creates the RayPerceptionSensor.
        /// </summary>
        /// <param name="name">The name of the sensor.</param>
        /// <param name="rayInput">The inputs for the sensor.</param>
        public RayPerceptionSensor(string name, RayPerceptionInput rayInput)
        {
            m_Name = name;
            m_RayPerceptionInput = rayInput;

            SetNumObservations(rayInput.OutputSize());

            if (Application.isEditor)
            {
                m_DebugDisplayInfo = new DebugDisplayInfo();
            }
        }

        void SetNumObservations(int numObservations)
        {
            m_Shape = new[] { numObservations };
            m_Observations = new float[numObservations];
        }

        internal void SetRayPerceptionInput(RayPerceptionInput rayInput)
        {
            // Note that change the number of rays or tags doesn't directly call this,
            // but changing them and then changing another field will.
            if (m_RayPerceptionInput.OutputSize() != rayInput.OutputSize())
            {
                Debug.Log(
                    "Changing the number of tags or rays at runtime is not " +
                    "supported and may cause errors in training or inference."
                );
                // Changing the shape will probably break things downstream, but we can at least
                // keep this consistent.
                SetNumObservations(rayInput.OutputSize());
            }
            m_RayPerceptionInput = rayInput;
        }

        /// <summary>
        /// Computes the ray perception observations and saves them to the provided
        /// <see cref="WriteAdapter"/>.
        /// </summary>
        /// <param name="adapter">Where the ray perception observations are written to.</param>
        /// <returns></returns>
        public int Write(WriteAdapter adapter)
        {
            using (TimerStack.Instance.Scoped("RayPerceptionSensor.Perceive"))
            {
                Array.Clear(m_Observations, 0, m_Observations.Length);

                var numRays = m_RayPerceptionInput.angles.Count;
                var numDetectableTags = m_RayPerceptionInput.detectableTags.Count;

                if (m_DebugDisplayInfo != null)
                {
                    // Reset the age information, and resize the buffer if needed.
                    m_DebugDisplayInfo.Reset();
                    if (m_DebugDisplayInfo.rayInfos == null || m_DebugDisplayInfo.rayInfos.Length != numRays)
                    {
                        m_DebugDisplayInfo.rayInfos = new DebugDisplayInfo.RayInfo[numRays];
                    }
                }

                // For each ray, do the casting, and write the information to the observation buffer
                for (var rayIndex = 0; rayIndex < numRays; rayIndex++)
                {
                    DebugDisplayInfo.RayInfo debugRay;
                    var rayOutput = PerceiveSingleRay(m_RayPerceptionInput, rayIndex, out debugRay);

                    if (m_DebugDisplayInfo != null)
                    {
                        m_DebugDisplayInfo.rayInfos[rayIndex] = debugRay;
                    }

                    rayOutput.ToFloatArray(numDetectableTags, rayIndex, m_Observations);
                }
                // Finally, add the observations to the WriteAdapter
                adapter.AddRange(m_Observations);
            }
            return m_Observations.Length;
        }

        /// <inheritdoc/>
        public void Update()
        {
        }

        /// <inheritdoc/>
        public void Reset() { }

        /// <inheritdoc/>
        public int[] GetObservationShape()
        {
            return m_Shape;
        }

        /// <inheritdoc/>
        public string GetName()
        {
            return m_Name;
        }

        /// <inheritdoc/>
        public virtual byte[] GetCompressedObservation()
        {
            return null;
        }

        /// <inheritdoc/>
        public virtual SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.None;
        }

        /// <summary>
        /// Evaluates the raycasts to be used as part of an observation of an agent.
        /// </summary>
        /// <param name="input">Input defining the rays that will be cast.</param>
        /// <returns>Output struct containing the raycast results.</returns>
        public static RayPerceptionOutput Perceive(RayPerceptionInput input)
        {
            RayPerceptionOutput output = new RayPerceptionOutput();
            output.rayOutputs = new RayPerceptionOutput.RayOutput[input.angles.Count];

            for (var rayIndex = 0; rayIndex < input.angles.Count; rayIndex++)
            {
                DebugDisplayInfo.RayInfo debugRay;
                output.rayOutputs[rayIndex] = PerceiveSingleRay(input, rayIndex, out debugRay);
            }

            return output;
        }

        /// <summary>
        /// Evaluate the raycast results of a single ray from the RayPerceptionInput.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="rayIndex"></param>
        /// <param name="debugRayOut"></param>
        /// <returns></returns>
        internal static RayPerceptionOutput.RayOutput PerceiveSingleRay(
            RayPerceptionInput input,
            int rayIndex,
            out DebugDisplayInfo.RayInfo debugRayOut
        )
        {
            var unscaledRayLength = input.rayLength;
            var unscaledCastRadius = input.castRadius;

            var extents = input.RayExtents(rayIndex);
            var startPositionWorld = extents.StartPositionWorld;
            var endPositionWorld = extents.EndPositionWorld;

            var rayDirection = endPositionWorld - startPositionWorld;
            // If there is non-unity scale, |rayDirection| will be different from rayLength.
            // We want to use this transformed ray length for determining cast length, hit fraction etc.
            // We also it to scale up or down the sphere or circle radii
            var scaledRayLength = rayDirection.magnitude;
            // Avoid 0/0 if unscaledRayLength is 0
            var scaledCastRadius = unscaledRayLength > 0 ?
                unscaledCastRadius * scaledRayLength / unscaledRayLength :
                unscaledCastRadius;

            // Do the cast and assign the hit information for each detectable tag.
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

            var rayOutput = new RayPerceptionOutput.RayOutput
            {
                hasHit = castHit,
                hitFraction = hitFraction,
                hitTaggedObject = false,
                hitTagIndex = -1
            };

            if (castHit)
            {
                // Find the index of the tag of the object that was hit.
                for (var i = 0; i < input.detectableTags.Count; i++)
                {
                    if (hitObject.CompareTag(input.detectableTags[i]))
                    {
                        rayOutput.hitTaggedObject = true;
                        rayOutput.hitTagIndex = i;
                        break;
                    }
                }
            }

            debugRayOut.worldStart = startPositionWorld;
            debugRayOut.worldEnd = endPositionWorld;
            debugRayOut.rayOutput = rayOutput;
            debugRayOut.castRadius = scaledCastRadius;

            return rayOutput;
        }
    }
}
