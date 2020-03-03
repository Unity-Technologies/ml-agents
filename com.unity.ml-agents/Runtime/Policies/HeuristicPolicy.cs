using System.Collections.Generic;
using System;
using System.Collections;
using MLAgents.Sensors;

namespace MLAgents.Policies
{
    /// <summary>
    /// The Heuristic Policy uses a hards coded Heuristic method
    /// to take decisions each time the RequestDecision method is
    /// called.
    /// </summary>
    internal class HeuristicPolicy : IPolicy
    {
        Func<float[]> m_Heuristic;
        float[] m_LastDecision;

        WriteAdapter m_WriteAdapter = new WriteAdapter();
        NullList m_NullList = new NullList();


        /// <inheritdoc />
        public HeuristicPolicy(Func<float[]> heuristic)
        {
            m_Heuristic = heuristic;
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors)
        {
            StepSensors(sensors);
            m_LastDecision = m_Heuristic.Invoke();
        }

        /// <inheritdoc />
        public float[] DecideAction()
        {
            return m_LastDecision;
        }

        public void Dispose()
        {
        }

        /// <summary>
        /// Trivial implementation of the IList interface that does nothing.
        /// This is only used for "writing" observations that we will discard.
        /// </summary>
        class NullList : IList<float>
        {
            public IEnumerator<float> GetEnumerator()
            {
                throw new NotImplementedException();
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }

            public void Add(float item)
            {
            }

            public void Clear()
            {
            }

            public bool Contains(float item)
            {
                return false;
            }

            public void CopyTo(float[] array, int arrayIndex)
            {
                throw new NotImplementedException();
            }

            public bool Remove(float item)
            {
                return false;
            }

            public int Count { get; }
            public bool IsReadOnly { get; }
            public int IndexOf(float item)
            {
                return -1;
            }

            public void Insert(int index, float item)
            {
            }

            public void RemoveAt(int index)
            {
            }

            public float this[int index]
            {
                get { return 0.0f; }
                set { }
            }
        }

        /// <summary>
        /// Run ISensor.Write or ISensor.GetCompressedObservation for each sensor
        /// The output is currently unused, but this makes the sensor usage consistent
        /// between training and inference.
        /// </summary>
        /// <param name="sensors"></param>
        void StepSensors(List<ISensor> sensors)
        {
            foreach (var sensor in sensors)
            {
                if (sensor.GetCompressionType() == SensorCompressionType.None)
                {
                    m_WriteAdapter.SetTarget(m_NullList, sensor.GetObservationShape(), 0);
                    sensor.Write(m_WriteAdapter);
                }
                else
                {
                    sensor.GetCompressedObservation();
                }
            }
        }
    }
}
