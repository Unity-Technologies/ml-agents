using System;
using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgents.Sensors
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
        public float RayLength;

        /// <summary>
        /// List of tags which correspond to object types agent can see.
        /// </summary>
        public IReadOnlyList<string> DetectableTags;

        /// <summary>
        /// List of angles (in degrees) used to define the rays.
        /// 90 degrees is considered "forward" relative to the game object.
        /// </summary>
        public IReadOnlyList<float> Angles;

        /// <summary>
        /// Starting height offset of ray from center of agent
        /// </summary>
        public float StartOffset;

        /// <summary>
        /// Ending height offset of ray from center of agent.
        /// </summary>
        public float EndOffset;

        /// <summary>
        /// Radius of the sphere to use for spherecasting.
        /// If 0 or less, rays are used instead - this may be faster, especially for complex environments.
        /// </summary>
        public float CastRadius;

        /// <summary>
        /// Transform of the GameObject.
        /// </summary>
        public Transform Transform;

        /// <summary>
        /// Whether to perform the casts in 2D or 3D.
        /// </summary>
        public RayPerceptionCastType CastType;

        /// <summary>
        /// Filtering options for the casts.
        /// </summary>
        public int LayerMask;

        /// <summary>
        /// Returns the expected number of floats in the output.
        /// </summary>
        /// <returns></returns>
        public int OutputSize()
        {
            return ((DetectableTags?.Count ?? 0) + 2) * (Angles?.Count ?? 0);
        }

        /// <summary>
        /// Get the cast start and end points for the given ray index/
        /// </summary>
        /// <param name="rayIndex"></param>
        /// <returns>A tuple of the start and end positions in world space.</returns>
        public (Vector3 StartPositionWorld, Vector3 EndPositionWorld) RayExtents(int rayIndex)
        {
            var angle = Angles[rayIndex];
            Vector3 startPositionLocal, endPositionLocal;
            if (CastType == RayPerceptionCastType.Cast3D)
            {
                startPositionLocal = new Vector3(0, StartOffset, 0);
                endPositionLocal = PolarToCartesian3D(RayLength, angle);
                endPositionLocal.y += EndOffset;
            }
            else
            {
                // Vector2s here get converted to Vector3s (and back to Vector2s for casting)
                startPositionLocal = new Vector2();
                endPositionLocal = PolarToCartesian2D(RayLength, angle);
            }

            var startPositionWorld = Transform.TransformPoint(startPositionLocal);
            var endPositionWorld = Transform.TransformPoint(endPositionLocal);

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
            public bool HasHit;

            /// <summary>
            /// Whether or not the ray hit an object whose tag is in the input's DetectableTags list.
            /// </summary>
            public bool HitTaggedObject;

            /// <summary>
            /// The index of the hit object's tag in the DetectableTags list, or -1 if there was no hit, or the
            /// hit object has a different tag.
            /// </summary>
            public int HitTagIndex;

            /// <summary>
            /// Normalized distance to the hit object.
            /// </summary>
            public float HitFraction;

            /// <summary>
            /// The hit GameObject (or null if there was no hit).
            /// </summary>
            public GameObject HitGameObject;

            /// <summary>
            /// Start position of the ray in world space.
            /// </summary>
            public Vector3 StartPositionWorld;

            /// <summary>
            /// End position of the ray in world space.
            /// </summary>
            public Vector3 EndPositionWorld;

            /// <summary>
            /// The scaled length of the ray.
            /// </summary>
            /// <remarks>
            /// If there is non-(1,1,1) scale, |EndPositionWorld - StartPositionWorld| will be different from
            /// the input rayLength.
            /// </remarks>
            public float ScaledRayLength
            {
                get
                {
                    var rayDirection = EndPositionWorld - StartPositionWorld;
                    return rayDirection.magnitude;
                }
            }

            /// <summary>
            /// The scaled size of the cast.
            /// </summary>
            /// <remarks>
            /// If there is non-(1,1,1) scale, the cast radius will be also be scaled.
            /// </remarks>
            public float ScaledCastRadius;

            /// <summary>
            /// Writes the ray output information to a subset of the float array.  Each element in the rayAngles array
            /// determines a sublist of data to the observation. The sublist contains the observation data for a single cast.
            /// The list is composed of the following:
            /// 1. A one-hot encoding for detectable tags. For example, if DetectableTags.Length = n, the
            ///    first n elements of the sublist will be a one-hot encoding of the detectableTag that was hit, or
            ///    all zeroes otherwise.
            /// 2. The 'numDetectableTags' element of the sublist will be 1 if the ray missed everything, or 0 if it hit
            ///    something (detectable or not).
            /// 3. The 'numDetectableTags+1' element of the sublist will contain the normalized distance to the object
            ///    hit, or 1.0 if nothing was hit.
            /// </summary>
            /// <param name="numDetectableTags"></param>
            /// <param name="rayIndex"></param>
            /// <param name="buffer">Output buffer. The size must be equal to (numDetectableTags+2) * RayOutputs.Length</param>
            public void ToFloatArray(int numDetectableTags, int rayIndex, float[] buffer)
            {
                var bufferOffset = (numDetectableTags + 2) * rayIndex;
                if (HitTaggedObject)
                {
                    buffer[bufferOffset + HitTagIndex] = 1f;
                }
                buffer[bufferOffset + numDetectableTags] = HasHit ? 0f : 1f;
                buffer[bufferOffset + numDetectableTags + 1] = HitFraction;
            }
        }

        /// <summary>
        /// RayOutput for each ray that was cast.
        /// </summary>
        public RayOutput[] RayOutputs;
    }

    /// <summary>
    /// A sensor implementation that supports ray cast-based observations.
    /// </summary>
    public class RayPerceptionSensor : ISensor, IBuiltInSensor
    {
        float[] m_Observations;
        ObservationSpec m_ObservationSpec;
        string m_Name;

        RayPerceptionInput m_RayPerceptionInput;
        RayPerceptionOutput m_RayPerceptionOutput;

        /// <summary>
        /// Time.frameCount at the last time Update() was called. This is only used for display in gizmos.
        /// </summary>
        int m_DebugLastFrameCount;

        internal int DebugLastFrameCount
        {
            get { return m_DebugLastFrameCount; }
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

            m_DebugLastFrameCount = Time.frameCount;
            m_RayPerceptionOutput = new RayPerceptionOutput();
        }

        /// <summary>
        /// The most recent raycast results.
        /// </summary>
        public RayPerceptionOutput RayPerceptionOutput
        {
            get { return m_RayPerceptionOutput; }
        }

        void SetNumObservations(int numObservations)
        {
            m_ObservationSpec = ObservationSpec.Vector(numObservations);
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
        /// <see cref="ObservationWriter"/>.
        /// </summary>
        /// <param name="writer">Where the ray perception observations are written to.</param>
        /// <returns></returns>
        public int Write(ObservationWriter writer)
        {
            using (TimerStack.Instance.Scoped("RayPerceptionSensor.Perceive"))
            {
                Array.Clear(m_Observations, 0, m_Observations.Length);
                var numRays = m_RayPerceptionInput.Angles.Count;
                var numDetectableTags = m_RayPerceptionInput.DetectableTags.Count;

                // For each ray, write the information to the observation buffer
                for (var rayIndex = 0; rayIndex < numRays; rayIndex++)
                {
                    m_RayPerceptionOutput.RayOutputs?[rayIndex].ToFloatArray(numDetectableTags, rayIndex, m_Observations);
                }

                // Finally, add the observations to the ObservationWriter
                writer.AddList(m_Observations);
            }
            return m_Observations.Length;
        }

        /// <inheritdoc/>
        public void Update()
        {
            m_DebugLastFrameCount = Time.frameCount;
            var numRays = m_RayPerceptionInput.Angles.Count;

            if (m_RayPerceptionOutput.RayOutputs == null || m_RayPerceptionOutput.RayOutputs.Length != numRays)
            {
                m_RayPerceptionOutput.RayOutputs = new RayPerceptionOutput.RayOutput[numRays];
            }

            // For each ray, do the casting and save the results.
            for (var rayIndex = 0; rayIndex < numRays; rayIndex++)
            {
                m_RayPerceptionOutput.RayOutputs[rayIndex] = PerceiveSingleRay(m_RayPerceptionInput, rayIndex);
            }
        }

        /// <inheritdoc/>
        public void Reset() { }

        /// <inheritdoc/>
        public ObservationSpec GetObservationSpec()
        {
            return m_ObservationSpec;
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
        public CompressionSpec GetCompressionSpec()
        {
            return CompressionSpec.Default();
        }

        /// <inheritdoc/>
        public BuiltInSensorType GetBuiltInSensorType()
        {
            return BuiltInSensorType.RayPerceptionSensor;
        }

        /// <summary>
        /// Evaluates the raycasts to be used as part of an observation of an agent.
        /// </summary>
        /// <param name="input">Input defining the rays that will be cast.</param>
        /// <returns>Output struct containing the raycast results.</returns>
        public static RayPerceptionOutput Perceive(RayPerceptionInput input)
        {
            RayPerceptionOutput output = new RayPerceptionOutput();
            output.RayOutputs = new RayPerceptionOutput.RayOutput[input.Angles.Count];

            for (var rayIndex = 0; rayIndex < input.Angles.Count; rayIndex++)
            {
                output.RayOutputs[rayIndex] = PerceiveSingleRay(input, rayIndex);
            }

            return output;
        }

        /// <summary>
        /// Evaluate the raycast results of a single ray from the RayPerceptionInput.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="rayIndex"></param>
        /// <returns></returns>
        internal static RayPerceptionOutput.RayOutput PerceiveSingleRay(
            RayPerceptionInput input,
            int rayIndex
        )
        {
            var unscaledRayLength = input.RayLength;
            var unscaledCastRadius = input.CastRadius;

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
            var castHit = false;
            var hitFraction = 1.0f;
            GameObject hitObject = null;

            if (input.CastType == RayPerceptionCastType.Cast3D)
            {
#if MLA_UNITY_PHYSICS_MODULE
                RaycastHit rayHit;
                if (scaledCastRadius > 0f)
                {
                    castHit = Physics.SphereCast(startPositionWorld, scaledCastRadius, rayDirection, out rayHit,
                        scaledRayLength, input.LayerMask);
                }
                else
                {
                    castHit = Physics.Raycast(startPositionWorld, rayDirection, out rayHit,
                        scaledRayLength, input.LayerMask);
                }

                // If scaledRayLength is 0, we still could have a hit with sphere casts (maybe?).
                // To avoid 0/0, set the fraction to 0.
                hitFraction = castHit ? (scaledRayLength > 0 ? rayHit.distance / scaledRayLength : 0.0f) : 1.0f;
                hitObject = castHit ? rayHit.collider.gameObject : null;
#endif
            }
            else
            {
#if MLA_UNITY_PHYSICS2D_MODULE
                RaycastHit2D rayHit;
                if (scaledCastRadius > 0f)
                {
                    rayHit = Physics2D.CircleCast(startPositionWorld, scaledCastRadius, rayDirection,
                        scaledRayLength, input.LayerMask);
                }
                else
                {
                    rayHit = Physics2D.Raycast(startPositionWorld, rayDirection, scaledRayLength, input.LayerMask);
                }

                castHit = rayHit;
                hitFraction = castHit ? rayHit.fraction : 1.0f;
                hitObject = castHit ? rayHit.collider.gameObject : null;
#endif
            }

            var rayOutput = new RayPerceptionOutput.RayOutput
            {
                HasHit = castHit,
                HitFraction = hitFraction,
                HitTaggedObject = false,
                HitTagIndex = -1,
                HitGameObject = hitObject,
                StartPositionWorld = startPositionWorld,
                EndPositionWorld = endPositionWorld,
                ScaledCastRadius = scaledCastRadius
            };

            if (castHit)
            {
                // Find the index of the tag of the object that was hit.
                var numTags = input.DetectableTags?.Count ?? 0;
                for (var i = 0; i < numTags; i++)
                {
                    var tagsEqual = false;
                    try
                    {
                        var tag = input.DetectableTags[i];
                        if (!string.IsNullOrEmpty(tag))
                        {
                            tagsEqual = hitObject.CompareTag(tag);
                        }
                    }
                    catch (UnityException)
                    {
                        // If the tag is null, empty, or not a valid tag, just ignore it.
                    }

                    if (tagsEqual)
                    {
                        rayOutput.HitTaggedObject = true;
                        rayOutput.HitTagIndex = i;
                        break;
                    }
                }
            }


            return rayOutput;
        }
    }
}
