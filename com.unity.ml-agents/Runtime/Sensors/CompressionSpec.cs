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
    }
}
