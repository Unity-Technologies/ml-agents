using System.Collections.Generic;
using UnityEngine;

namespace MLAgents.Sensor
{
    public class VectorSensor : SensorBase
    {
        public List<float> observations;
        int[] m_Shape;
        string m_Name;

        public VectorSensor(int observationSize, List<float> observations = null, string name = null)
        {
            if (name == null)
            {
                name = $"VectorSensor_size{observationSize}";
            }

            if (observations == null)
            {
                observations = new List<float>(observationSize);
            }

            this.observations = observations;
            m_Name = name;
            m_Shape = new[] { observationSize };

        }

        public override void WriteObservation(float[] output)
        {
            if (observations.Count > output.Length)
            {
                Debug.LogWarningFormat(
                    "Too many observations ({0} > {1}). Only the first {1} will be used",
                    observations.Count, output.Length
                );
            }

            // TODO warn if not enough?

            var size = Mathf.Min(observations.Count, output.Length);
            for (var i = 0; i < size; i++)
            {
                output[i] = observations[i];
            }
        }

        public override int[] GetFloatObservationShape()
        {
            return m_Shape;
        }

        public override string GetName()
        {
            return m_Name;
        }

        public override void Update()
        {
            observations.Clear();
        }
    }
}
