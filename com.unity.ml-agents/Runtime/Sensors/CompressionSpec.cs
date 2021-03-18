using System.Linq;
namespace Unity.MLAgents.Sensors
{
    public struct CompressionSpec
    {
        internal SensorCompressionType m_SensorCompressionType;

        public SensorCompressionType SensorCompressionType
        {
            get => m_SensorCompressionType;
        }

        internal int[] m_CompressedChannelMapping;

        /// The mapping of the channels in compressed data to the actual channel after decompression.
        /// The mapping is a list of integer index with the same length as
        /// the number of output observation layers (channels), including padding if there's any.
        /// Each index indicates the actual channel the layer will go into.
        /// Layers with the same index will be averaged, and layers with negative index will be dropped.
        /// For example, mapping for CameraSensor using grayscale and stacking of two: [0, 0, 0, 1, 1, 1]
        /// Mapping for GridSensor of 4 channels and stacking of two: [0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1, -1]
        public int[] CompressedChannelMapping
        {
            get => m_CompressedChannelMapping;
        }

        public static CompressionSpec Default()
        {
            return new CompressionSpec
            {
                SensorCompressionType = SensorCompressionType.None,
                CompressedChannelMapping = null
            };
        }

        public static CompressionSpec Compressed(SensorCompressionType sensorCompressionType, int[] compressedChannelMapping = null)
        {
            return new CompressionSpec
            {
                SensorCompressionType = sensorCompressionType,
                CompressedChannelMapping = compressedChannelMapping
            };
        }

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
