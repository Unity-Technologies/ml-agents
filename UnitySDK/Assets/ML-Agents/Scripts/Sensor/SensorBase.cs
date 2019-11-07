using UnityEngine;

namespace MLAgents.Sensor
{
    public abstract class SensorBase : ISensor
    {
        /// <summary>
        /// Write the observations to the output buffer. This size of the buffer will be product of the sizes returned
        /// by GetFloatObservationShape().
        /// </summary>
        /// <param name="output"></param>
        public abstract void WriteObservation(float[] output);

        public abstract int[] GetFloatObservationShape();

        public abstract string GetName();

        /// <summary>
        /// Default implementation of Write interface. This creates a temporary array, calls WriteObservation,
        /// and then writes the results to the WriteAdapter.
        /// </summary>
        /// <param name="adapter"></param>
        public virtual int Write(WriteAdapter adapter)
        {
            // TODO reuse buffer for similar agents, don't call GetFloatObservationShape()
            var numFloats = this.ObservationSize();
            float[] buffer = new float[numFloats];
            WriteObservation(buffer);

            adapter.AddRange(buffer);

            return numFloats;
        }

        public void Update() { }

        public virtual byte[] GetCompressedObservation()
        {
            return null;
        }

        public virtual SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.None;
        }
    }
}
