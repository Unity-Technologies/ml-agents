using System.Collections.Generic;
using UnityEngine;

namespace MLAgents.Sensors
{
    /// <summary>
    /// A sensor implementation for vector observations.
    /// </summary>
    public class VectorSensor : ISensor
    {
        // TODO use float[] instead
        // TODO allow setting float[]
        List<float> m_Observations;
        int[] m_Shape;
        string m_Name;

        /// <summary>
        /// Initializes the sensor.
        /// </summary>
        /// <param name="observationSize">Number of vector observations.</param>
        /// <param name="name">Name of the sensor.</param>
        public VectorSensor(int observationSize, string name = null)
        {
            if (name == null)
            {
                name = $"VectorSensor_size{observationSize}";
            }

            m_Observations = new List<float>(observationSize);
            m_Name = name;
            m_Shape = new[] { observationSize };
        }

        /// <inheritdoc/>
        public int Write(WriteAdapter adapter)
        {
            var expectedObservations = m_Shape[0];
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
            adapter.AddRange(m_Observations);
            return expectedObservations;
        }

        /// <inheritdoc/>
        public void Update()
        {
            Clear();
        }

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

        void Clear()
        {
            m_Observations.Clear();
        }

        void AddFloatObs(float obs)
        {
#if DEBUG
            Utilities.DebugCheckNanAndInfinity(obs, nameof(obs), nameof(AddFloatObs));
#endif
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
        /// Adds a collection of float observations to the vector observations of the agent.
        /// </summary>
        /// <param name="observation">Observation.</param>
        public void AddObservation(IEnumerable<float> observation)
        {
            foreach (var f in observation)
            {
                AddFloatObs(f);
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
        /// <param name="range">The max index for any observation.</param>
        public void AddOneHotObservation(int observation, int range)
        {
            for (var i = 0; i < range; i++)
            {
                AddFloatObs(i == observation ? 1.0f : 0.0f);
            }
        }
    }
}
