using MLAgents.InferenceBrain;

namespace MLAgents.Sensor
{
    public enum CompressionType
    {
        None,
        PNG,
        Custom1
    }

    /// <summary>
    /// Sensor interface for generating observations.
    /// For custom implementations, it is recommended to SensorBase instead.
    /// </summary>
    public interface ISensor {
        /// <summary>
        /// Returns the size of the observations that will be generated.
        /// For example, a sensor that observes the velocity of a rigid body (in 3D) would return new {3}.
        /// A sensor that returns an RGB image would return new [] {Width, Height, 3}
        /// </summary>
        /// <returns></returns>
        int[] GetFloatObservationShape();

        /// <summary>
        /// Write the observation data directly to the TensorProxy.
        /// This is considered an advanced interface; for a simpler apporach, use SensorBase and override WriteFloats instead.
        /// </summary>
        /// <param name="proxy"></param>
        /// <param name="index"></param>
        void WriteToTensor(TensorProxy tensorProxy, int agentIndex);

        /// <summary>
        /// Return a compressed representation of the observation. For small observations, this sbould generally not be
        /// implemented. However, compressing large observations (such as visual results) can significantly improve
        /// model training time.
        /// </summary>
        /// <returns></returns>
        byte[] GetCompressedObservation();

        /// <summary>
        /// Return the compression type being used. If no compression is used, return CompressionType.None
        /// </summary>
        /// <returns></returns>
        CompressionType GetCompressionType();
    }

}
