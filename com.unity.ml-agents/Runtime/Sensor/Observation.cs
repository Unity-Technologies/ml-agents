using System;
using UnityEngine;

namespace MLAgents.Sensor
{
    public struct Observation
    {
        /// <summary>
        /// The compressed sensor data. Assumed to be non-null if CompressionType != CompressionType.None
        /// </summary>
        public byte[] CompressedData;

        /// <summary>
        /// Uncompressed sensor data. Assumed to be non-empty if CompressionType == CompressionType.None
        /// </summary>
        public ArraySegment<float> FloatData;

        /// <summary>
        /// The format of the compressed data
        /// </summary>
        public SensorCompressionType CompressionType;

        /// <summary>
        /// The uncompressed dimensions of the data.
        /// E.g. for RGB visual observations, this will be {Width, Height, 3}
        /// </summary>
        public int[] Shape;
    }
}
