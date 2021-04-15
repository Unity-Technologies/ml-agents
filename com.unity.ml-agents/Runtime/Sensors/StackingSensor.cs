using System;
using System.Linq;
using UnityEngine;
using Unity.Barracuda;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// Sensor that wraps around another Sensor to provide temporal stacking.
    /// Conceptually, consecutive observations are stored left-to-right, which is how they're output
    /// For example, 4 stacked sets of observations would be output like
    ///   |  t = now - 3  |  t = now -3  |  t = now - 2  |  t = now  |
    /// Internally, a circular buffer of arrays is used. The m_CurrentIndex represents the most recent observation.
    /// Currently, observations are stacked on the last dimension.
    /// </summary>
    public class StackingSensor : ISensor, IBuiltInSensor
    {
        /// <summary>
        /// The wrapped sensor.
        /// </summary>
        ISensor m_WrappedSensor;

        /// <summary>
        /// Number of stacks to save
        /// </summary>
        int m_NumStackedObservations;
        int m_UnstackedObservationSize;

        string m_Name;
        private ObservationSpec m_ObservationSpec;
        private ObservationSpec m_WrappedSpec;

        /// <summary>
        /// Buffer of previous observations
        /// </summary>
        float[][] m_StackedObservations;

        byte[][] m_StackedCompressedObservations;

        int m_CurrentIndex;
        ObservationWriter m_LocalWriter = new ObservationWriter();

        byte[] m_EmptyCompressedObservation;
        int[] m_CompressionMapping;
        TensorShape m_tensorShape;

        /// <summary>
        /// Initializes the sensor.
        /// </summary>
        /// <param name="wrapped">The wrapped sensor.</param>
        /// <param name="numStackedObservations">Number of stacked observations to keep.</param>
        public StackingSensor(ISensor wrapped, int numStackedObservations)
        {
            // TODO ensure numStackedObservations > 1
            m_WrappedSensor = wrapped;
            m_NumStackedObservations = numStackedObservations;

            m_Name = $"StackingSensor_size{numStackedObservations}_{wrapped.GetName()}";

            m_WrappedSpec = wrapped.GetObservationSpec();

            m_UnstackedObservationSize = wrapped.ObservationSize();

            // Set up the cached observation spec for the StackingSensor
            var newShape = m_WrappedSpec.Shape;
            // TODO support arbitrary stacking dimension
            newShape[newShape.Length - 1] *= numStackedObservations;
            m_ObservationSpec = new ObservationSpec(
                newShape, m_WrappedSpec.DimensionProperties, m_WrappedSpec.ObservationType
            );

            // Initialize uncompressed buffer anyway in case python trainer does not
            // support the compression mapping and has to fall back to uncompressed obs.
            m_StackedObservations = new float[numStackedObservations][];
            for (var i = 0; i < numStackedObservations; i++)
            {
                m_StackedObservations[i] = new float[m_UnstackedObservationSize];
            }

            if (m_WrappedSensor.GetCompressionSpec().SensorCompressionType != SensorCompressionType.None)
            {
                m_StackedCompressedObservations = new byte[numStackedObservations][];
                m_EmptyCompressedObservation = CreateEmptyPNG();
                for (var i = 0; i < numStackedObservations; i++)
                {
                    m_StackedCompressedObservations[i] = m_EmptyCompressedObservation;
                }
                m_CompressionMapping = ConstructStackedCompressedChannelMapping(wrapped);
            }

            if (m_WrappedSpec.Rank != 1)
            {
                var wrappedShape = m_WrappedSpec.Shape;
                m_tensorShape = new TensorShape(0, wrappedShape[0], wrappedShape[1], wrappedShape[2]);
            }
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            // First, call the wrapped sensor's write method. Make sure to use our own writer, not the passed one.
            m_LocalWriter.SetTarget(m_StackedObservations[m_CurrentIndex], m_WrappedSpec, 0);
            m_WrappedSensor.Write(m_LocalWriter);

            // Now write the saved observations (oldest first)
            var numWritten = 0;
            if (m_WrappedSpec.Rank == 1)
            {
                for (var i = 0; i < m_NumStackedObservations; i++)
                {
                    var obsIndex = (m_CurrentIndex + 1 + i) % m_NumStackedObservations;
                    writer.AddList(m_StackedObservations[obsIndex], numWritten);
                    numWritten += m_UnstackedObservationSize;
                }
            }
            else
            {
                for (var i = 0; i < m_NumStackedObservations; i++)
                {
                    var obsIndex = (m_CurrentIndex + 1 + i) % m_NumStackedObservations;
                    for (var h = 0; h < m_WrappedSpec.Shape[0]; h++)
                    {
                        for (var w = 0; w < m_WrappedSpec.Shape[1]; w++)
                        {
                            for (var c = 0; c < m_WrappedSpec.Shape[2]; c++)
                            {
                                writer[h, w, i * m_WrappedSpec.Shape[2] + c] = m_StackedObservations[obsIndex][m_tensorShape.Index(0, h, w, c)];
                            }
                        }
                    }
                }
                numWritten = m_WrappedSpec.Shape[0] * m_WrappedSpec.Shape[1] * m_WrappedSpec.Shape[2] * m_NumStackedObservations;
            }

            return numWritten;
        }

        /// <summary>
        /// Updates the index of the "current" buffer.
        /// </summary>
        public void Update()
        {
            m_WrappedSensor.Update();
            m_CurrentIndex = (m_CurrentIndex + 1) % m_NumStackedObservations;
        }

        /// <inheritdoc/>
        public void Reset()
        {
            m_WrappedSensor.Reset();
            // Zero out the buffer.
            for (var i = 0; i < m_NumStackedObservations; i++)
            {
                Array.Clear(m_StackedObservations[i], 0, m_StackedObservations[i].Length);
            }
            if (m_WrappedSensor.GetCompressionSpec().SensorCompressionType != SensorCompressionType.None)
            {
                for (var i = 0; i < m_NumStackedObservations; i++)
                {
                    m_StackedCompressedObservations[i] = m_EmptyCompressedObservation;
                }
            }
        }

        /// <inheritdoc/>
        public ObservationSpec GetObservationSpec()
        {
            return m_ObservationSpec;
        }

        /// <inheritdoc/>
        public string GetName()
        {
            return m_Name;
        }

        /// <inheritdoc/>
        public byte[] GetCompressedObservation()
        {
            var compressed = m_WrappedSensor.GetCompressedObservation();
            m_StackedCompressedObservations[m_CurrentIndex] = compressed;

            int bytesLength = 0;
            foreach (byte[] compressedObs in m_StackedCompressedObservations)
            {
                bytesLength += compressedObs.Length;
            }

            byte[] outputBytes = new byte[bytesLength];
            int offset = 0;
            for (var i = 0; i < m_NumStackedObservations; i++)
            {
                var obsIndex = (m_CurrentIndex + 1 + i) % m_NumStackedObservations;
                Buffer.BlockCopy(m_StackedCompressedObservations[obsIndex],
                    0, outputBytes, offset, m_StackedCompressedObservations[obsIndex].Length);
                offset += m_StackedCompressedObservations[obsIndex].Length;
            }

            return outputBytes;
        }

        /// <inheritdoc/>
        public CompressionSpec GetCompressionSpec()
        {
            var wrappedSpec = m_WrappedSensor.GetCompressionSpec();
            return new CompressionSpec(wrappedSpec.SensorCompressionType, m_CompressionMapping);
        }

        /// <summary>
        /// Create Empty PNG for initializing the buffer for stacking.
        /// </summary>
        internal byte[] CreateEmptyPNG()
        {
            var shape = m_WrappedSpec.Shape;
            int height = shape[0];
            int width = shape[1];
            var texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);
            Color32[] resetColorArray = texture2D.GetPixels32();
            Color32 black = new Color32(0, 0, 0, 0);
            for (int i = 0; i < resetColorArray.Length; i++)
            {
                resetColorArray[i] = black;
            }
            texture2D.SetPixels32(resetColorArray);
            texture2D.Apply();
            return texture2D.EncodeToPNG();
        }

        /// <summary>
        /// Construct stacked CompressedChannelMapping.
        /// </summary>
        internal int[] ConstructStackedCompressedChannelMapping(ISensor wrappedSenesor)
        {
            // Get CompressedChannelMapping of the wrapped sensor. If the
            // wrapped sensor doesn't have one, use default mapping.
            // Default mapping: {0, 0, 0} for grayscale, identity mapping {1, 2, ..., n} otherwise.
            int[] wrappedMapping = null;
            int wrappedNumChannel = m_WrappedSpec.Shape[2];

            wrappedMapping = wrappedSenesor.GetCompressionSpec().CompressedChannelMapping;
            if (wrappedMapping == null)
            {
                if (wrappedNumChannel == 1)
                {
                    wrappedMapping = new[] { 0, 0, 0 };
                }
                else
                {
                    wrappedMapping = Enumerable.Range(0, wrappedNumChannel).ToArray();
                }
            }

            // Construct stacked mapping using the mapping of wrapped sensor.
            // First pad the wrapped mapping to multiple of 3, then repeat
            // and add offset to each copy to form the stacked mapping.
            int paddedMapLength = (wrappedMapping.Length + 2) / 3 * 3;
            var compressionMapping = new int[paddedMapLength * m_NumStackedObservations];
            for (var i = 0; i < m_NumStackedObservations; i++)
            {
                var offset = wrappedNumChannel * i;
                for (var j = 0; j < paddedMapLength; j++)
                {
                    if (j < wrappedMapping.Length)
                    {
                        compressionMapping[j + paddedMapLength * i] = wrappedMapping[j] >= 0 ? wrappedMapping[j] + offset : -1;
                    }
                    else
                    {
                        compressionMapping[j + paddedMapLength * i] = -1;
                    }
                }
            }
            return compressionMapping;
        }

        /// <inheritdoc/>
        public BuiltInSensorType GetBuiltInSensorType()
        {
            IBuiltInSensor wrappedBuiltInSensor = m_WrappedSensor as IBuiltInSensor;
            return wrappedBuiltInSensor?.GetBuiltInSensorType() ?? BuiltInSensorType.Unknown;
        }
    }
}
