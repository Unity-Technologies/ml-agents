using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// The way the GridSensor process detected colliders in a cell.
    /// </summary>
    public enum ProcessCollidersMethod
    {
        /// <summary>
        /// Get data from all colliders detected in a cell
        /// </summary>
        ProcessAllColliders,

        /// <summary>
        /// Get data from the collider closest to the agent
        /// </summary>
        ProcessClosestColliders
    }

    /// <summary>
    /// Grid-based sensor.
    /// </summary>
    public class GridSensorBase : ISensor, IBuiltInSensor, IDisposable
    {
        string m_Name;
        Vector3 m_CellScale;
        Vector3Int m_GridSize;
        string[] m_DetectableTags;
        SensorCompressionType m_CompressionType;
        ObservationSpec m_ObservationSpec;
        internal IGridPerception m_GridPerception;

        // Buffers
        float[] m_PerceptionBuffer;
        Color[] m_PerceptionColors;
        Texture2D m_PerceptionTexture;
        float[] m_CellDataBuffer;

        // Utility Constants Calculated on Init
        int m_NumCells;
        int m_CellObservationSize;
        Vector3 m_CellCenterOffset;


        /// <summary>
        /// Create a GridSensorBase with the specified configuration.
        /// </summary>
        /// <param name="name">The sensor name</param>
        /// <param name="cellScale">The scale of each cell in the grid</param>
        /// <param name="gridSize">Number of cells on each side of the grid</param>
        /// <param name="detectableTags">Tags to be detected by the sensor</param>
        /// <param name="compression">Compression type</param>
        public GridSensorBase(
            string name,
            Vector3 cellScale,
            Vector3Int gridSize,
            string[] detectableTags,
            SensorCompressionType compression
        )
        {
            m_Name = name;
            m_CellScale = cellScale;
            m_GridSize = gridSize;
            m_DetectableTags = detectableTags;
            CompressionType = compression;

            if (m_GridSize.y != 1)
            {
                throw new UnityAgentsException("GridSensor only supports 2D grids.");
            }

            m_NumCells = m_GridSize.x * m_GridSize.z;
            m_CellObservationSize = GetCellObservationSize();
            m_ObservationSpec = ObservationSpec.Visual(m_CellObservationSize, m_GridSize.x, m_GridSize.z);
            m_PerceptionTexture = new Texture2D(m_GridSize.x, m_GridSize.z, TextureFormat.RGB24, false);

            ResetPerceptionBuffer();
        }

        /// <summary>
        /// The compression type used by the sensor.
        /// </summary>
        public SensorCompressionType CompressionType
        {
            get { return m_CompressionType; }
            set
            {
                if (!IsDataNormalized() && value == SensorCompressionType.PNG)
                {
                    Debug.LogWarning($"Compression type {value} is only supported with normalized data. " +
                        "The sensor will not compress the data.");
                    return;
                }
                m_CompressionType = value;
            }
        }

        internal float[] PerceptionBuffer
        {
            get { return m_PerceptionBuffer; }
        }

        /// <summary>
        /// The tags which the sensor dectects.
        /// </summary>
        protected string[] DetectableTags
        {
            get { return m_DetectableTags; }
        }

        /// <inheritdoc/>
        public void Reset() { }

        /// <summary>
        /// Clears the perception buffer before loading in new data.
        /// </summary>
        public void ResetPerceptionBuffer()
        {
            if (m_PerceptionBuffer != null)
            {
                Array.Clear(m_PerceptionBuffer, 0, m_PerceptionBuffer.Length);
                Array.Clear(m_CellDataBuffer, 0, m_CellDataBuffer.Length);
            }
            else
            {
                m_PerceptionBuffer = new float[m_CellObservationSize * m_NumCells];
                m_CellDataBuffer = new float[m_CellObservationSize];
                m_PerceptionColors = new Color[m_NumCells];
            }
        }

        /// <inheritdoc/>
        public string GetName()
        {
            return m_Name;
        }

        /// <inheritdoc/>
        public CompressionSpec GetCompressionSpec()
        {
            return new CompressionSpec(CompressionType);
        }

        /// <inheritdoc/>
        public BuiltInSensorType GetBuiltInSensorType()
        {
            return BuiltInSensorType.GridSensor;
        }

        /// <inheritdoc/>
        public byte[] GetCompressedObservation()
        {
            using (TimerStack.Instance.Scoped("GridSensor.GetCompressedObservation"))
            {
                var allBytes = new List<byte>();
                var numImages = (m_CellObservationSize + 2) / 3;
                for (int i = 0; i < numImages; i++)
                {
                    var channelIndex = 3 * i;
                    GridValuesToTexture(channelIndex, Math.Min(3, m_CellObservationSize - channelIndex));
                    allBytes.AddRange(m_PerceptionTexture.EncodeToPNG());
                }

                return allBytes.ToArray();
            }
        }

        /// <summary>
        /// Convert observation values to texture for PNG compression.
        /// </summary>
        void GridValuesToTexture(int channelIndex, int numChannelsToAdd)
        {
            for (int i = 0; i < m_NumCells; i++)
            {
                for (int j = 0; j < numChannelsToAdd; j++)
                {
                    m_PerceptionColors[i][j] = m_PerceptionBuffer[i * m_CellObservationSize + channelIndex + j];
                }
            }
            m_PerceptionTexture.SetPixels(m_PerceptionColors);
        }

        /// <summary>
        /// Get the observation values of the detected game object.
        /// Default is to record the detected tag index.
        ///
        /// This method can be overridden to encode the observation differently or get custom data from the object.
        /// When overriding this method, <seealso cref="GetCellObservationSize"/> and <seealso cref="IsDataNormalized"/>
        /// might also need to change accordingly.
        /// </summary>
        /// <param name="detectedObject">The game object that was detected within a certain cell</param>
        /// <param name="tagIndex">The index of the detectedObject's tag in the DetectableObjects list</param>
        /// <param name="dataBuffer">The buffer to write the observation values.
        ///         The buffer size is configured by <seealso cref="GetCellObservationSize"/>.
        /// </param>
        /// <example>
        ///   Here is an example of overriding GetObjectData to get the velocity of a potential Rigidbody:
        ///   <code>
        ///     protected override void GetObjectData(GameObject detectedObject, int tagIndex, float[] dataBuffer)
        ///     {
        ///         if (tagIndex == Array.IndexOf(DetectableTags, "RigidBodyObject"))
        ///         {
        ///             Rigidbody rigidbody = detectedObject.GetComponent&lt;Rigidbody&gt;();
        ///             dataBuffer[0] = rigidbody.velocity.x;
        ///             dataBuffer[1] = rigidbody.velocity.y;
        ///             dataBuffer[2] = rigidbody.velocity.z;
        ///         }
        ///     }
        ///  </code>
        /// </example>
        protected virtual void GetObjectData(GameObject detectedObject, int tagIndex, float[] dataBuffer)
        {
            dataBuffer[0] = tagIndex + 1;
        }

        /// <summary>
        /// Get the observation size for each cell. This will be the size of dataBuffer for <seealso cref="GetObjectData"/>.
        /// If overriding <seealso cref="GetObjectData"/>, override this method as well to the custom observation size.
        /// </summary>
        /// <returns>The observation size of each cell.</returns>
        protected virtual int GetCellObservationSize()
        {
            return 1;
        }

        /// <summary>
        /// Whether the data is normalized within [0, 1]. The sensor can only use PNG compression if the data is normailzed.
        /// If overriding <seealso cref="GetObjectData"/>, override this method as well according to the custom observation values.
        /// </summary>
        /// <returns>Bool value indicating whether data is normalized.</returns>
        protected virtual bool IsDataNormalized()
        {
            return false;
        }

        /// <summary>
        /// Whether to process all detected colliders in a cell. Default to false and only use the one closest to the agent.
        /// If overriding <seealso cref="GetObjectData"/>, consider override this method when needed.
        /// </summary>
        /// <returns>Bool value indicating whether to process all detected colliders in a cell.</returns>
        protected internal virtual ProcessCollidersMethod GetProcessCollidersMethod()
        {
            return ProcessCollidersMethod.ProcessClosestColliders;
        }

        /// <summary>
        /// If using PNG compression, check if the values are normalized.
        /// </summary>
        void ValidateValues(float[] dataValues, GameObject detectedObject)
        {
            if (m_CompressionType != SensorCompressionType.PNG)
            {
                return;
            }

            for (int j = 0; j < dataValues.Length; j++)
            {
                if (dataValues[j] < 0 || dataValues[j] > 1)
                    throw new UnityAgentsException($"When using compression type {m_CompressionType} the data value has to be normalized between 0-1. " +
                        $"Received value[{dataValues[j]}] for {detectedObject.name}");
            }
        }

        /// <summary>
        /// Collect data from the detected object if a detectable tag is matched.
        /// </summary>
        internal void ProcessDetectedObject(GameObject detectedObject, int cellIndex)
        {
            Profiler.BeginSample("GridSensor.ProcessDetectedObject");
            for (var i = 0; i < m_DetectableTags.Length; i++)
            {
                if (!ReferenceEquals(detectedObject, null) && detectedObject.CompareTag(m_DetectableTags[i]))
                {
                    if (GetProcessCollidersMethod() == ProcessCollidersMethod.ProcessAllColliders)
                    {
                        Array.Copy(m_PerceptionBuffer, cellIndex * m_CellObservationSize, m_CellDataBuffer, 0, m_CellObservationSize);
                    }
                    else
                    {
                        Array.Clear(m_CellDataBuffer, 0, m_CellDataBuffer.Length);
                    }

                    GetObjectData(detectedObject, i, m_CellDataBuffer);
                    ValidateValues(m_CellDataBuffer, detectedObject);
                    Array.Copy(m_CellDataBuffer, 0, m_PerceptionBuffer, cellIndex * m_CellObservationSize, m_CellObservationSize);
                    break;
                }
            }
            Profiler.EndSample();
        }

        /// <inheritdoc/>
        public void Update()
        {
            ResetPerceptionBuffer();
            using (TimerStack.Instance.Scoped("GridSensor.Update"))
            {
                if (m_GridPerception != null)
                {
                    m_GridPerception.Perceive();
                }
            }
        }

        /// <inheritdoc/>
        public ObservationSpec GetObservationSpec()
        {
            return m_ObservationSpec;
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            using (TimerStack.Instance.Scoped("GridSensor.Write"))
            {
                int index = 0;
                for (var h = m_GridSize.z - 1; h >= 0; h--)
                {
                    for (var w = 0; w < m_GridSize.x; w++)
                    {
                        for (var d = 0; d < m_CellObservationSize; d++)
                        {
                            writer[d, h, w] = m_PerceptionBuffer[index];
                            index++;
                        }
                    }
                }
                return index;
            }
        }

        /// <summary>
        /// Clean up the internal objects.
        /// </summary>
        public void Dispose()
        {
            if (!ReferenceEquals(null, m_PerceptionTexture))
            {
                Utilities.DestroyTexture(m_PerceptionTexture);
                m_PerceptionTexture = null;
            }
        }
    }
}
