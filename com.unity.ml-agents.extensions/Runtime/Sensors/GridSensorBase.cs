using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;
using Unity.MLAgents.Sensors;
using UnityEngine.Profiling;
using Object = UnityEngine.Object;

[assembly: InternalsVisibleTo("Unity.ML-Agents.Extensions.TestUtils")]
namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Grid-based sensor.
    /// </summary>
    public class GridSensorBase : ISensor, IBuiltInSensor, IDisposable
    {
        protected string m_Name;
        Vector3 m_CellScale;
        Vector3Int m_GridSize;
        string[] m_DetectableObjects;
        SensorCompressionType m_CompressionType;
        ObservationSpec m_ObservationSpec;
        internal BoxOverlapChecker m_BoxOverlapChecker;

        // Buffers
        float[] m_PerceptionBuffer;
        Color[] m_PerceptionColors;
        Texture2D m_PerceptionTexture;
        float[] m_CellDataBuffer;

        // Utility Constants Calculated on Init
        int m_NumCells;
        int m_CellObservationSize;
        Vector3 m_CellCenterOffset;


        public GridSensorBase(
            string name,
            Vector3 cellScale,
            Vector3Int gridNum,
            string[] detectableObjects,
            SensorCompressionType compression
        )
        {
            m_Name = name;
            m_CellScale = cellScale;
            m_GridSize = gridNum;
            m_DetectableObjects = detectableObjects;
            m_CompressionType = compression;

            if (m_GridSize.y != 1)
            {
                throw new UnityAgentsException("GridSensor only supports 2D grids.");
            }

            m_NumCells = m_GridSize.x * m_GridSize.z;
            m_CellObservationSize = GetCellObservationSize();
            m_ObservationSpec = ObservationSpec.Visual(m_GridSize.x, m_GridSize.z, m_CellObservationSize);
            m_PerceptionTexture = new Texture2D(m_GridSize.x, m_GridSize.z, TextureFormat.RGB24, false);

            ResetPerceptionBuffer();
        }

        public SensorCompressionType CompressionType
        {
            get { return m_CompressionType; }
            set
            {
                if (!IsDataNormalized() && value == SensorCompressionType.PNG)
                {
                    Debug.LogWarning("Compression type {m_CompressionType} is only supported with normalized data. " +
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

        protected string[] DetectableObjects
        {
            get { return m_DetectableObjects; }
        }

        /// <inheritdoc/>
        public void Reset() { }

        /// <summary>
        /// Clears the perception buffer before loading in new data. If the gridDepthType is ChannelHot, then it initializes the
        /// Reset() also reinits the cell activity array (for debug)
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

        /// <summary>
        /// GetCompressedObservation - Calls Perceive then puts the data stored on the perception buffer
        /// onto the m_perceptionTexture2D to be converted to a byte array and returned
        /// </summary>
        /// <returns>byte[] containing the compressed observation of the grid observation</returns>
        public byte[] GetCompressedObservation()
        {
            using (TimerStack.Instance.Scoped("GridSensor.GetCompressedObservation"))
            {
                var allBytes = new List<byte>();
                var numImages = (m_CellObservationSize + 2) / 3;
                for (int i = 0; i < numImages; i++)
                {
                    var channelIndex = 3 * i;
                    ChannelsToTexture(channelIndex, Math.Min(3, m_CellObservationSize - channelIndex));
                    allBytes.AddRange(m_PerceptionTexture.EncodeToPNG());
                }

                return allBytes.ToArray();
            }
        }

        /// <summary>
        /// ChannelsToTexture - Takes the channel index and the numChannelsToAdd.
        /// For each cell and for each channel to add, sets it to a value of the color specified for that cell.
        ///  All colors are then set to the perceptionTexture via SetPixels.
        /// m_perceptionTexture2D can then be read as an image as it now contains all of the information that was
        /// stored in the channels
        /// </summary>
        /// <param name="channelIndex"></param>
        /// <param name="numChannelsToAdd"></param>
        void ChannelsToTexture(int channelIndex, int numChannelsToAdd)
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
        /// GetObjectData - returns an array of values that represent the game object
        /// This is one of the few methods that one may need to override to get their required functionality
        /// For instance, if one wants specific information about the current gameobject, they can use this method
        /// to extract it and then return it in an array format.
        /// </summary>
        /// <returns>
        /// A float[] containing the data that holds the representative information of the passed in gameObject
        /// </returns>
        /// <param name="currentColliderGo">The game object that was found colliding with a certain cell</param>
        /// <param name="typeIndex">The index of the type (tag) of the gameObject.
        ///           (e.g., if this GameObject had the 3rd tag out of 4, type_index would be 2.0f)</param>
        /// <param name="normalizedDistance">A float between 0 and 1 describing the ratio of
        ///            the distance currentColliderGo is compared to the edge of the gridsensor</param>
        /// <example>
        ///   Here is an example of extenind GetObjectData to include information about a potential Rigidbody:
        ///   <code>
        ///     protected override float[] GetObjectData(GameObject currentColliderGo,
        ///                                     float type_index, float normalized_distance)
        ///     {
        ///         float[] channelValues = new float[ChannelDepth.Length]; // ChannelDepth.Length = 4 in this example
        ///         channelValues[0] = type_index;
        ///         Rigidbody goRb = currentColliderGo.GetComponent&lt;Rigidbody&gt;();
        ///         if (goRb != null)
        ///         {
        ///             channelValues[1] = goRb.velocity.x;
        ///             channelValues[2] = goRb.velocity.y;
        ///             channelValues[3] = goRb.velocity.z;
        ///         }
        ///         return channelValues;
        ///     }
        ///  </code>
        /// </example>
        protected virtual void GetObjectData(GameObject currentColliderGo, int typeIndex, float[] dataBuffer)
        {
            dataBuffer[0] = typeIndex;
        }

        /// <summary>
        /// Get the observation size for each cell. This will be the size of dataBuffer in GetObjectData().
        /// </summary>
        protected virtual int GetCellObservationSize()
        {
            return 1;
        }

        /// <summary>
        /// Whether the data is normailzed within [0, 1]. The sensor can only use PNG compression if the data is normailzed.
        /// </summary>
        protected virtual bool IsDataNormalized()
        {
            return false;
        }

        /// <summary>
        /// Whether to process all the colliders detected in a cell. Default to false and only use the one closest to th agent.
        /// </summary>
        protected internal virtual bool ProcessAllCollidersInCell()
        {
            return false;
        }

        /// <summary>
        /// Runs basic validation assertions to check that the values can be normalized
        /// </summary>
        /// <param name="channelValues">The values to be validated</param>
        /// <param name="currentColliderGo">The gameobject used for better error messages</param>
        protected virtual void ValidateValues(float[] channelValues, GameObject currentColliderGo)
        {
            if (m_CompressionType != SensorCompressionType.PNG)
            {
                return;
            }

            for (int j = 0; j < channelValues.Length; j++)
            {
                if (channelValues[j] < 0 || channelValues[j] > 1)
                    throw new UnityAgentsException($"When using compression type {m_CompressionType} the data value has to be normalized between 0-1. " +
                        $"Received value[{channelValues[j]}] for {currentColliderGo.name}");
            }
        }

        /// <summary>
        /// LoadObjectData - If the GameObject matches a tag, GetObjectData is called to extract the data from the GameObject
        /// then the data is transformed based on the GridDepthType of the gridsensor.
        /// Further documetation on the GridDepthType can be found below
        /// </summary>
        /// <param name="currentColliderGo">The game object that was found colliding with a certain cell</param>
        /// <param name="cellIndex">The index of the current cell</param>
        protected internal void LoadObjectData(GameObject currentColliderGo, int cellIndex)
        {
            Profiler.BeginSample("GridSensor.LoadObjectData");
            for (var i = 0; i < m_DetectableObjects.Length; i++)
            {
                if (!ReferenceEquals(currentColliderGo, null) && currentColliderGo.CompareTag(m_DetectableObjects[i]))
                {
                    Array.Clear(m_CellDataBuffer, 0, m_CellDataBuffer.Length);
                    GetObjectData(currentColliderGo, i, m_CellDataBuffer);
                    ValidateValues(m_CellDataBuffer, currentColliderGo);
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
                if (m_BoxOverlapChecker != null)
                {
                    m_BoxOverlapChecker.Update();
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
                            writer[h, w, d] = m_PerceptionBuffer[index];
                            index++;
                        }
                    }
                }
                return index;
            }
        }

        public void Dispose()
        {
            if (!ReferenceEquals(null, m_PerceptionTexture))
            {
                DestroyTexture(m_PerceptionTexture);
                m_PerceptionTexture = null;
            }
        }
        static void DestroyTexture(Texture2D texture)
        {
            if (Application.isEditor)
            {
                // Edit Mode tests complain if we use Destroy()
                // TODO move to extension methods for UnityEngine.Object?
                Object.DestroyImmediate(texture);
            }
            else
            {
                Object.Destroy(texture);
            }
        }
    }
}
