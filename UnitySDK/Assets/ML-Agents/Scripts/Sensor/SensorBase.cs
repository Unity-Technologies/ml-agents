using MLAgents.InferenceBrain;

namespace MLAgents.Sensor
{
    abstract class SensorBase : ISensor
    {
        /// <summary>
        /// Write the observations to the output buffer. This size of the buffer will be product of the sizes returned
        /// by GetFloatObservationShape().
        /// </summary>
        /// <param name="output"></param>
        public abstract void WriteObservation(float[] output);

        public abstract int[] GetFloatObservationShape();

        /// <summary>
        /// Default implementation of WriteToTensor interface. This creates a temporary array, calls WriteObservation,
        /// and then writes the results to the TensorProxy.
        /// </summary>
        /// <param name="tensorProxy"></param>
        /// <param name="index"></param>
        public void WriteToTensor(TensorProxy tensorProxy, int index)
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
                tensorProxy.data[index, i] = buffer[i];
            }
        }

        public byte[] GetCompressedObservation()
        {
            return null;
        }

        public CompressionType GetCompressionType()
        {
            return CompressionType.None;
        }
    }
}
