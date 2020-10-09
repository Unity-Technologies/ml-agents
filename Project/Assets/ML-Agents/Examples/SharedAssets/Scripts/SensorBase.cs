using Unity.MLAgents.Sensors;

namespace Unity.MLAgentsExamples
{
    /// <summary>
    /// A simple sensor that provides a number default implementations.
    /// </summary>
    public abstract class SensorBase : ISensor
    {
        /// <summary>
        /// Write the observations to the output buffer. This size of the buffer will be product
        /// of the sizes returned by <see cref="GetObservationShape"/>.
        /// </summary>
        /// <param name="output"></param>
        public abstract void WriteObservation(float[] output);

        /// <inheritdoc/>
        public abstract int[] GetObservationShape();

        /// <inheritdoc/>
        public abstract string GetName();

        /// <summary>
        /// Default implementation of Write interface. This creates a temporary array,
        /// calls WriteObservation, and then writes the results to the ObservationWriter.
        /// </summary>
        /// <param name="writer"></param>
        /// <returns>The number of elements written.</returns>
        public virtual int Write(ObservationWriter writer)
        {
            // TODO reuse buffer for similar agents, don't call GetObservationShape()
            var numFloats = this.ObservationSize();
            float[] buffer = new float[numFloats];
            WriteObservation(buffer);

            writer.AddRange(buffer);

            return numFloats;
        }

        /// <inheritdoc/>
        public void Update() { }

        /// <inheritdoc/>
        public void Reset() { }

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
    }
}
