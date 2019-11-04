using MLAgents.InferenceBrain;
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
        /// Default implementation of WriteToTensor interface. This creates a temporary array, calls WriteObservation,
        /// and then writes the results to the TensorProxy.
        /// </summary>
        /// <param name="tensorProxy"></param>
        /// <param name="agentIndex"></param>
        public virtual void WriteToTensor(TensorProxy tensorProxy, int agentIndex)
        {
            // TODO reuse buffer for similar agents, don't call GetFloatObservationShape()
            int[] shape = GetFloatObservationShape();
            int numFloats = 1;
            foreach (var dim in shape)
            {
                numFloats *= dim;
            }

            float[] buffer = new float[numFloats];
            WriteObservation(buffer);

            for (var i = 0; i < numFloats; i++)
            {
                tensorProxy.data[agentIndex, i] = buffer[i];
            }
        }

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
