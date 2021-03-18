using System.Linq;
namespace Unity.MLAgents.Sensors
{
    public struct CompressionSpec
    {
        public SensorCompressionType SensorCompressionType;
        public int[] CompressedChannelMapping;

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
