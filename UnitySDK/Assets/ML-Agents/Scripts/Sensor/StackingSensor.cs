using System.Collections.Generic;
using UnityEngine.Assertions.Comparers;

namespace MLAgents.Sensor
{
    /// <summary>
    /// Sensor that wraps around another Sensor to provide temporal stacking.
    /// </summary>
    public class StackingSensor : SensorBase
    {
        /// <summary>
        /// The wrapped sensor.
        /// </summary>
        SensorBase m_WrappedSensor;

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
        // TODO Flat list like old code? Circular buffer?
        List<float> m_StackedObservations;

        /// <summary>
        ///
        /// </summary>
        /// <param name="wrapped">The wrapped sensor</param>
        /// <param name="numStackedObservations">Number of stacked observations to keep</param>
        public StackingSensor(SensorBase wrapped, int numStackedObservations)
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
            m_StackedObservations = new List<float>(m_NumUnstackedObservationSize * m_NumStackedObservations);
        }

        public override void WriteObservation(float[] output)
        {
            // TODO optimize - keep a temp version or use a circular buffer of float[]s
            var tempOut = new float[m_NumUnstackedObservationSize];
            m_WrappedSensor.WriteObservation(tempOut);

            Utilities.ReplaceRange(m_StackedObservations, tempOut,m_StackedObservations.Count - tempOut.Length);

            // Write all stacked to output
            for (var i = 0; i < m_StackedObservations.Count; i++)
            {
                output[i] = m_StackedObservations[i];
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
            Utilities.ShiftLeft(m_StackedObservations, m_NumUnstackedObservationSize);

            m_WrappedSensor.Update();
        }

        // TODO support stacked compressed observations (byte stream)

    }
}
