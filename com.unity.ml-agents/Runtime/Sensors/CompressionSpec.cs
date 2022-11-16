using System.Linq;
namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// The compression setting for visual/camera observations.
    /// </summary>
    public enum SensorCompressionType
    {
        /// <summary>
        /// No compression. Data is preserved as float arrays.
        /// </summary>
        None,

        /// <summary>
        /// PNG format. Data will be stored in binary format.
        /// </summary>
        PNG
    }

    /// <summary>
    /// A description of the compression used for observations.
    /// </summary>
    /// <remarks>
    /// Most ISensor implementations can't take advantage of compression,
    /// and should return CompressionSpec.Default() from their ISensor.GetCompressionSpec() methods.
    /// Visual observations, or mulitdimensional categorical observations (for example, image segmentation
    /// or the piece types in a match-3 game board) can use PNG compression reduce the amount of
    /// data transferred between Unity and the trainer.
    /// </remarks>
    public struct CompressionSpec
    {
        internal SensorCompressionType m_SensorCompressionType;

        /// <summary>
        /// The compression type that the sensor will use for its observations.
        /// </summary>
        public SensorCompressionType SensorCompressionType
        {
            get => m_SensorCompressionType;
        }

        internal int[] m_CompressedChannelMapping;

        /// <summary>
        /// The mapping of the channels in compressed data to the actual channel after decompression.
        /// </summary>
        /// <remarks>
        /// The mapping is a list of integer index with the same length as
        /// the number of output observation layers (channels), including padding if there's any.
        /// Each index indicates the actual channel the layer will go into.
        /// Layers with the same index will be averaged, and layers with negative index will be dropped.
        /// For example, mapping for CameraSensor using grayscale and stacking of two: [0, 0, 0, 1, 1, 1]
        /// Mapping for GridSensor of 4 channels and stacking of two: [0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1, -1]
        /// </remarks>
        public int[] CompressedChannelMapping
        {
            get => m_CompressedChannelMapping;
        }

        /// <summary>
        /// Return a CompressionSpec indicating possible compression.
        /// </summary>
        /// <param name="sensorCompressionType">The compression type to use.</param>
        /// <param name="compressedChannelMapping">Optional mapping mapping of the channels in compressed data to the
        /// actual channel after decompression.</param>
        public CompressionSpec(SensorCompressionType sensorCompressionType, int[] compressedChannelMapping = null)
        {
            m_SensorCompressionType = sensorCompressionType;
            m_CompressedChannelMapping = compressedChannelMapping;
        }

        /// <summary>
        /// Return a CompressionSpec indicating no compression. This is recommended for most sensors.
        /// </summary>
        /// <returns></returns>
        public static CompressionSpec Default()
        {
            return new CompressionSpec
            {
                m_SensorCompressionType = SensorCompressionType.None,
                m_CompressedChannelMapping = null
            };
        }

        /// <summary>
        /// Return whether the compressed channel mapping is "trivial"; if so it doesn't need to be sent to the
        /// trainer.
        /// </summary>
        /// <returns></returns>
        internal bool IsTrivialMapping()
        {
            var mapping = CompressedChannelMapping;
            if (mapping == null)
            {
                return true;
            }
            // check if mapping equals zero mapping
            if (mapping.Length == 3 && mapping.All(m => m == 0))
            {
                return true;
            }
            // check if mapping equals identity mapping
            for (var i = 0; i < mapping.Length; i++)
            {
                if (mapping[i] != i)
                {
                    return false;
                }
            }
            return true;
        }
    }
}
