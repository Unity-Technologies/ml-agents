using System.Collections.Generic;

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
        int m_StackSize;

        int m_StackingDimension;
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
        /// <param name="stackSize">Number of stacked observations to keep</param>
        /// <param name="stackingDimension">The dimension of the observation to stack on.
        ///     -1 implies the last dimension.
        /// </param>
        public StackingSensor(SensorBase wrapped, int stackSize, int stackingDimension=-1)
        {
            m_WrappedSensor = wrapped;
            m_StackSize = stackSize;

            m_Name = wrapped.GetName() + "_stacked";

            var shape = wrapped.GetFloatObservationShape();
            m_Shape = new int[shape.Length];
            m_StackingDimension = stackingDimension < 0 ? shape.Length - 1 : stackingDimension;


            for (int d = 0; d < shape.Length; d++)
            {
                m_Shape[d] = shape[d];
            }

            m_Shape[m_StackingDimension] *= 2;
        }

        public override void WriteObservation(float[] output)
        {
            // TODO
            // Update m_StackedObservations (or add separate call for that)
            // Call m_wrapped.Write
            // Write all stacked to output
        }

        public override int[] GetFloatObservationShape()
        {
            return m_Shape;
        }

        public override string GetName()
        {
            return m_Name;
        }

        // TODO support stacked compressed observations (byte stream)

    }
}
