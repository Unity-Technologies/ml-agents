using System.Collections.Generic;
using System.Collections.ObjectModel;
using UnityEngine;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// A sensor implementation for vector observations.
    /// </summary>
    public class VectorSensor : ISensor, IBuiltInSensor
    {
        // TODO use float[] instead
        // TODO allow setting float[]
        List<float> m_Observations;
        ObservationSpec m_ObservationSpec;
        string m_Name;

        /// <summary>
        /// Initializes the sensor.
        /// </summary>
        /// <param name="observationSize">Number of vector observations.</param>
        /// <param name="name">Name of the sensor.</param>
        /// <param name="observationType"></param>
        public VectorSensor(int observationSize, string name = null, ObservationType observationType = ObservationType.Default)
        {
            if (string.IsNullOrEmpty(name))
            {
                name = $"VectorSensor_size{observationSize}";
                if (observationType != ObservationType.Default)
                {
                    name += $"_{observationType.ToString()}";
                }
            }

            m_Observations = new List<float>(observationSize);
            m_Name = name;
            m_ObservationSpec = ObservationSpec.Vector(observationSize, observationType);
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            var expectedObservations = m_ObservationSpec.Shape[0];
            if (m_Observations.Count > expectedObservations)
            {
                // Too many observations, truncate
                Debug.LogWarningFormat(
                    "More observations ({0}) made than vector observation size ({1}). The observations will be truncated.",
                    m_Observations.Count, expectedObservations
                );
                m_Observations.RemoveRange(expectedObservations, m_Observations.Count - expectedObservations);
            }
            else if (m_Observations.Count < expectedObservations)
            {
                // Not enough observations; pad with zeros.
                Debug.LogWarningFormat(
                    "Fewer observations ({0}) made than vector observation size ({1}). The observations will be padded.",
                    m_Observations.Count, expectedObservations
                );
                for (int i = m_Observations.Count; i < expectedObservations; i++)
                {
                    m_Observations.Add(0);
                }
            }
            writer.AddList(m_Observations);
            return expectedObservations;
        }

        /// <summary>
        /// Returns a read-only view of the observations that added.
        /// </summary>
        /// <returns>A read-only view of the observations list.</returns>
        internal ReadOnlyCollection<float> GetObservations()
        {
            return m_Observations.AsReadOnly();
        }

        /// <inheritdoc/>
        public void Update()
        {
            Clear();
        }

        /// <inheritdoc/>
        public void Reset()
        {
            Clear();
        }

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
            return BuiltInSensorType.VectorSensor;
        }

        void Clear()
        {
            m_Observations.Clear();
        }

        void AddFloatObs(float obs)
        {
            Utilities.DebugCheckNanAndInfinity(obs, nameof(obs), nameof(AddFloatObs));
            m_Observations.Add(obs);
        }

        // Compatibility methods with Agent observation. These should be removed eventually.

        /// <summary>
        /// Adds a float observation to the vector observations of the agent.
        /// </summary>
        /// <param name="observation">Observation.</param>
        public void AddObservation(float observation)
        {
            AddFloatObs(observation);
        }

        /// <summary>
        /// Adds an integer observation to the vector observations of the agent.
        /// </summary>
        /// <param name="observation">Observation.</param>
        public void AddObservation(int observation)
        {
            AddFloatObs(observation);
        }

        /// <summary>
        /// Adds an Vector3 observation to the vector observations of the agent.
        /// </summary>
        /// <param name="observation">Observation.</param>
        public void AddObservation(Vector3 observation)
        {
            AddFloatObs(observation.x);
            AddFloatObs(observation.y);
            AddFloatObs(observation.z);
        }

        /// <summary>
        /// Adds an Vector2 observation to the vector observations of the agent.
        /// </summary>
        /// <param name="observation">Observation.</param>
        public void AddObservation(Vector2 observation)
        {
            AddFloatObs(observation.x);
            AddFloatObs(observation.y);
        }

        /// <summary>
        /// Adds a list or array of float observations to the vector observations of the agent.
        /// </summary>
        /// <param name="observation">Observation.</param>
        public void AddObservation(IList<float> observation)
        {
            for (var i = 0; i < observation.Count; i++)
            {
                AddFloatObs(observation[i]);
            }
        }

        /// <summary>
        /// Adds a quaternion observation to the vector observations of the agent.
        /// </summary>
        /// <param name="observation">Observation.</param>
        public void AddObservation(Quaternion observation)
        {
            AddFloatObs(observation.x);
            AddFloatObs(observation.y);
            AddFloatObs(observation.z);
            AddFloatObs(observation.w);
        }

        /// <summary>
        /// Adds a boolean observation to the vector observation of the agent.
        /// </summary>
        /// <param name="observation">Observation.</param>
        public void AddObservation(bool observation)
        {
            AddFloatObs(observation ? 1f : 0f);
        }

        /// <summary>
        /// Adds a one-hot encoding observation.
        /// </summary>
        /// <param name="observation">The index of this observation.</param>
        /// <param name="range">The upper limit on the value observation can take (exclusive).</param>
        public void AddOneHotObservation(int observation, int range)
        {
            for (var i = 0; i < range; i++)
            {
                AddFloatObs(i == observation ? 1.0f : 0.0f);
            }
        }
    }
}
