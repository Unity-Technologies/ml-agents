using System.Collections.Generic;
using UnityEngine.Assertions.Comparers;

namespace MLAgents.Sensor
{
    /// <summary>
    /// Sensor that wraps around another Sensor to provide temporal stacking.
    /// Conceptually, consecutive observations are stored left-to-right, which is how they're output
    /// For example, 4 stacked sets of observations would be output like
    ///   |  t = now - 3  |  t = now -3  |  t = now - 2  |  t = now  |
    /// Internally, a circular buffer of arrays is used. The m_CurrentIndex represents the most recent observation.
    /// </summary>
    public class StackingSensor : ISensor
    {
        /// <summary>
        /// The wrapped sensor.
        /// </summary>
        ISensor m_WrappedSensor;

        /// <summary>
        /// Number of stacks to save
        /// </summary>
        int m_NumStackedObservations;
        int m_NumUnstackedObservationSize;

        string m_Name;
        int[] m_Shape;

        /// <summary>
        /// Buffer of previous observations
        /// </summary>
        float[][] m_StackedObservations;

        int m_currentIndex;
        WriteAdapter m_LocalAdapter = new WriteAdapter();

        /// <summary>
        ///
        /// </summary>
        /// <param name="wrapped">The wrapped sensor</param>
        /// <param name="numStackedObservations">Number of stacked observations to keep</param>
        public StackingSensor(ISensor wrapped, int numStackedObservations)
        {
            // TODO ensure numStackedObservations > 1
            m_WrappedSensor = wrapped;
            m_NumStackedObservations = numStackedObservations;

            m_Name = wrapped.GetName() + "_stacked";

            var shape = wrapped.GetFloatObservationShape();
            m_Shape = new int[shape.Length];

            m_NumUnstackedObservationSize = 1;
            for (int d = 0; d < shape.Length; d++)
            {
                m_Shape[d] = shape[d];
                m_NumUnstackedObservationSize *= shape[d];
            }

            // TODO support arbitrary stacking dimension
            m_Shape[0] *= numStackedObservations;
            m_StackedObservations = new float[numStackedObservations][];
            for (var i = 0; i < numStackedObservations; i++)
            {
                m_StackedObservations[i] = new float[m_NumUnstackedObservationSize];
            }
        }

        public int Write(WriteAdapter adapter)
        {
            // First, call the wrapped sensor's write method. Make sure to use our own adapater, not the passed one.
            m_LocalAdapter.SetTarget(m_StackedObservations[m_currentIndex], 0);
            m_WrappedSensor.Write(m_LocalAdapter);

            // Now write the saved observations (oldest first)
            var numWritten = 0;
            for (var i = 0; i < m_NumStackedObservations; i++)
            {
                var obsIndex = (m_currentIndex + 1 + i) % m_NumStackedObservations;
                adapter.AddRange(m_StackedObservations[obsIndex], numWritten);
                numWritten += m_NumUnstackedObservationSize;
            }

            // Finally update the index of the "current" buffer.
            m_currentIndex = (m_currentIndex + 1) % m_NumStackedObservations;
            return numWritten;
        }

        public int[] GetFloatObservationShape()
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

        // TODO support stacked compressed observations (byte stream)

    }
}
