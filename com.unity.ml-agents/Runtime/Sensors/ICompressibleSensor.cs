namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// Sensor interface for generating observations.
    /// </summary>
    public interface ICompressibleSensor : ISensor
    {
        /// <summary>
        /// Returns the mapping of the the channels of compressed bytes to the
        /// actual channel after decompression.
        /// </summary>
        /// <returns>Mapping of the compressed data</returns>
        int[] GetCompressionMapping();
    }
}
