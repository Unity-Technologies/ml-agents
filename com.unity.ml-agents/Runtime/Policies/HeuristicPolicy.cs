using System.Collections.Generic;
using System;
using System.Collections;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Policies
{
    /// <summary>
    /// The Heuristic Policy uses a hard-coded Heuristic method
    /// to take decisions each time the RequestDecision method is
    /// called.
    /// </summary>
    internal class HeuristicPolicy : IPolicy
    {
        ActuatorManager m_ActuatorManager;
        ActionBuffers m_ActionBuffers;
        bool m_Done;
        bool m_DecisionRequested;

        ObservationWriter m_ObservationWriter = new ObservationWriter();
        NullList m_NullList = new NullList();


        public HeuristicPolicy(ActuatorManager actuatorManager, ActionSpec actionSpec)
        {
            m_ActuatorManager = actuatorManager;
            var numContinuousActions = actionSpec.NumContinuousActions;
            var numDiscreteActions = actionSpec.NumDiscreteActions;
            var continuousDecision = new ActionSegment<float>(new float[numContinuousActions], 0, numContinuousActions);
            var discreteDecision = new ActionSegment<int>(new int[numDiscreteActions], 0, numDiscreteActions);
            m_ActionBuffers = new ActionBuffers(continuousDecision, discreteDecision);
        }

        /// <inheritdoc />
        public void RequestDecision(AgentInfo info, List<ISensor> sensors)
        {
            StepSensors(sensors);
            m_Done = info.done;
            m_DecisionRequested = true;
        }

        /// <inheritdoc />
        public ref readonly ActionBuffers DecideAction()
        {
            if (!m_Done && m_DecisionRequested)
            {
                m_ActionBuffers.Clear();
                m_ActuatorManager.ApplyHeuristic(m_ActionBuffers);
            }
            m_DecisionRequested = false;
            return ref m_ActionBuffers;
        }

        public void Dispose()
        {
        }

        /// <summary>
        /// Trivial implementation of the IList interface that does nothing.
        /// This is only used for "writing" observations that we will discard.
        /// </summary>
        internal class NullList : IList<float>
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
                if (sensor.GetCompressionSpec().SensorCompressionType == SensorCompressionType.None)
                {
                    m_ObservationWriter.SetTarget(m_NullList, sensor.GetObservationSpec(), 0);
                    sensor.Write(m_ObservationWriter);
                }
                else
                {
                    sensor.GetCompressedObservation();
                }
            }
        }
    }
}
