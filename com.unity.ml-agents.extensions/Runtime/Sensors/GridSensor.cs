using System;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.MLAgents.Sensors;
using UnityEngine.Profiling;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Enum describing what kind of depth type the data should be organized as
    /// </summary>
    public enum GridDepthType { Channel, ChannelHot, Counting };

    /// <summary>
    /// Grid-based sensor.
    /// </summary>
    [AddComponentMenu("ML Agents/Grid Sensor", (int)MenuGroup.Sensors)]
    public class GridSensor : ISensor, IBuiltInSensor
    {
        // Sensor parameters configured in UI
        string m_Name;
        float m_CellScaleX;
        float m_CellScaleY;
        float m_CellScaleZ;
        int m_GridNumSideX;
        int m_GridNumSideZ;
        bool m_RotateWithAgent;
        int[] m_ChannelDepth;
        string[] m_DetectableObjects;
        LayerMask m_ObserveMask;
        GridDepthType m_GridDepthType;
        GameObject m_RootReference;
        int m_MaxColliderBufferSize;
        int m_InitialColliderBufferSize;
        SensorCompressionType m_CompressionType;
        bool m_ShowGizmos;
        Color[] m_DebugColors;

        // Buffers and intermediate objects
        // The buffer to store all collider found
        Collider[] m_ColliderBuffer;
        // The starting index of each channel in one-hot representation for channel-hot type
        float[] m_ChannelBuffer;
        // The offsets used to specify where within a cell's allotted data, certain observations will be inserted.
        int[] m_ChannelOffsets;
        // The main storage of perceptual information.
        float[] m_PerceptionBuffer;
        // The default value of the perceptionBuffer when using the ChannelHot DepthType. Used to reset the array.
        float[] m_ChannelHotDefaultPerceptionBuffer;
        // Array of Colors needed in order to load the values of the perception buffer to a texture.
        Color[] m_PerceptionColors;
        // Array of index of colors displaying the DebugColors for each cell in OnDrawGizmos. Only updated if ShowGizmos.
        int[] m_CellActivity;
        // Array of positions where each position is the center of a cell.
        Vector3[] m_CellPoints;
        // Texture where the colors are written to so that they can be compressed in PNG format.
        Texture2D m_Texture2D;

        // Utility constants calculated on init
        // Total Number of cells (width*height)
        int m_NumCells;
        // The total number of observations per cell of the grid. Its equivalent to the "channel" on the outgoing tensor.
        int m_ObservationPerCell;
        // Radius of grid, used for normalizing the distance.
        float m_InverseSphereRadius;
        //  (gridNumSideZ - 1) / 2; Offset used for calculating CellToPoint
        float m_OffsetGridNumSide;
        // Cached ObservationSpec
        ObservationSpec m_ObservationSpec;


        public GridSensor(
            string name,
            float cellScaleX,
            float cellScaleY,
            float cellScaleZ,
            int gridNumSideX,
            int gridNumSideZ,
            bool rotateWithAgent,
            int[] channelDepth,
            string[] detectableObjects,
            LayerMask observeMask,
            GridDepthType gridDepthType,
            GameObject rootReference,
            int maxColliderBufferSize,
            int initialColliderBufferSize,
            Color[] debugColors,
            bool showGizmos,
            SensorCompressionType compression
        )
        {
            m_Name = name;
            m_CellScaleX = cellScaleX;
            m_CellScaleY = cellScaleY;
            m_CellScaleZ = cellScaleZ;
            m_GridNumSideX = gridNumSideX;
            m_GridNumSideZ = gridNumSideZ;
            m_RotateWithAgent = rotateWithAgent;
            m_ChannelDepth = channelDepth;
            m_DetectableObjects = detectableObjects;
            m_ObserveMask = observeMask;
            m_GridDepthType = gridDepthType;
            m_RootReference = rootReference;
            m_MaxColliderBufferSize = maxColliderBufferSize;
            m_InitialColliderBufferSize = initialColliderBufferSize;
            m_DebugColors = debugColors;
            m_ShowGizmos = showGizmos;
            m_CompressionType = compression;

            if (m_GridDepthType == GridDepthType.Counting && m_DetectableObjects.Length != m_ChannelDepth.Length)
            {
                throw new UnityAgentsException("The channels of a CountingGridSensor is equal to the number of detectableObjects");
            }

            m_ObservationSpec = ObservationSpec.Visual(m_GridNumSideX, m_GridNumSideZ, m_ObservationPerCell);
            m_Texture2D = new Texture2D(m_GridNumSideX, m_GridNumSideZ, TextureFormat.RGB24, false);
            m_ColliderBuffer = new Collider[Math.Min(m_MaxColliderBufferSize, m_InitialColliderBufferSize)];

            InitGridParameters();
            InitDepthType();
            InitCellPoints();
            ResetPerceptionBuffer();
        }

        public SensorCompressionType CompressionType
        {
            get { return m_CompressionType; }
            set { m_CompressionType = value;}
        }

        public bool ShowGizmos
        {
            get { return m_ShowGizmos; }
            set { m_ShowGizmos = value;}
        }

        public Color[] DebugColors
        {
            get { return m_DebugColors; }
            set { m_DebugColors = value;}
        }

        public int[] CellActivity
        {
            get { return m_CellActivity; }
        }

        /// <summary>
        /// Initializes the constant parameters used within the perceive method call
        /// </summary>
        void InitGridParameters()
        {
            m_NumCells = m_GridNumSideX * m_GridNumSideZ;
            float sphereRadiusX = (m_CellScaleX * m_GridNumSideX) / Mathf.Sqrt(2);
            float sphereRadiusZ = (m_CellScaleZ * m_GridNumSideZ) / Mathf.Sqrt(2);
            m_InverseSphereRadius = 1.0f / Mathf.Max(sphereRadiusX, sphereRadiusZ);
            m_OffsetGridNumSide = (m_GridNumSideZ - 1f) / 2f;
        }

        /// <summary>
        /// Initializes the constant parameters that are based on the Grid Depth Type
        /// Sets the ObservationPerCell and the ChannelOffsets properties
        /// </summary>
        void InitDepthType()
        {
            if (m_GridDepthType == GridDepthType.ChannelHot)
            {
                m_ObservationPerCell = m_ChannelDepth.Sum();

                m_ChannelOffsets = new int[m_ChannelDepth.Length];
                for (int i = 1; i < m_ChannelDepth.Length; i++)
                {
                    m_ChannelOffsets[i] = m_ChannelOffsets[i - 1] + m_ChannelDepth[i - 1];
                }

                m_ChannelHotDefaultPerceptionBuffer = new float[m_ObservationPerCell];
                for (int i = 0; i < m_ChannelDepth.Length; i++)
                {
                    if (m_ChannelDepth[i] > 1)
                    {
                        m_ChannelHotDefaultPerceptionBuffer[m_ChannelOffsets[i]] = 1;
                    }
                }
            }
            else
            {
                m_ObservationPerCell = m_ChannelDepth.Length;
            }

            // The maximum number of channels in the final output must be less than 255 * 3 because the "number of PNG images" to generate must fit in one byte
            Assert.IsTrue(m_ObservationPerCell < (255 * 3), "The maximum number of channels per cell must be less than 255 * 3");
        }

        /// <summary>
        /// Initializes the location of the CellPoints property
        /// </summary>
        void InitCellPoints()
        {
            m_CellPoints = new Vector3[m_NumCells];

            for (int i = 0; i < m_NumCells; i++)
            {
                m_CellPoints[i] = CellToPoint(i);
            }
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
                if (m_GridDepthType == GridDepthType.ChannelHot)
                {
                    // Copy the default value to the array
                    for (int i = 0; i < m_NumCells; i++)
                    {
                        Array.Copy(m_ChannelHotDefaultPerceptionBuffer, 0, m_PerceptionBuffer, i * m_ObservationPerCell, m_ObservationPerCell);
                    }
                }
                else
                {
                    Array.Clear(m_PerceptionBuffer, 0, m_PerceptionBuffer.Length);
                }
            }
            else
            {
                m_PerceptionBuffer = new float[m_ObservationPerCell * m_NumCells];
                m_ColliderBuffer = new Collider[Math.Min(m_MaxColliderBufferSize, m_InitialColliderBufferSize)];
                m_ChannelBuffer = new float[m_ChannelDepth.Length];
                m_PerceptionColors = new Color[m_NumCells];
            }

            if (m_ShowGizmos)
            {
                // Ensure to init arrays if not yet assigned (for editor)
                if (m_CellActivity == null)
                    m_CellActivity = new int[m_NumCells];

                // Assign the default color to the cell activities
                for (int i = 0; i < m_NumCells; i++)
                {
                    m_CellActivity[i] = -1;
                }
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
            return new CompressionSpec(m_CompressionType);
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
                Perceive(); // Fill the perception buffer with observed data

                var allBytes = new List<byte>();
                var numImages = (m_ObservationPerCell + 2) / 3;
                for (int i = 0; i < numImages; i++)
                {
                    var channelIndex = 3 * i;
                    ChannelsToTexture(channelIndex, Math.Min(3, m_ObservationPerCell - channelIndex));
                    allBytes.AddRange(m_Texture2D.EncodeToPNG());
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
                    m_PerceptionColors[i][j] = m_PerceptionBuffer[i * m_ObservationPerCell + channelIndex + j];
                }
            }
            m_Texture2D.SetPixels(m_PerceptionColors);
        }

        /// <summary>
        /// Perceive - Clears the buffers, calls overlap box on the actual cell (the actual perception part)
        /// for all found colliders, LoadObjectData is called
        /// at the end, Perceive returns the float array of the perceptions
        /// </summary>
        /// <returns>A float[] containing all of the information collected from the gridsensor</returns>
        internal void Perceive()
        {
            if (m_ColliderBuffer == null)
            {
                return;
            }

            ResetPerceptionBuffer();
            using (TimerStack.Instance.Scoped("GridSensor.Perceive"))
            {
                var halfCellScale = new Vector3(m_CellScaleX / 2f, m_CellScaleY, m_CellScaleZ / 2f);

                for (var cellIndex = 0; cellIndex < m_NumCells; cellIndex++)
                {
                    var cellCenter = GetCellGlobalPosition(cellIndex);
                    var numFound = BufferResizingOverlapBoxNonAlloc(cellCenter, halfCellScale, GetGridRotation());

                    if (numFound > 0)
                    {
                        if (m_GridDepthType == GridDepthType.Counting)
                        {
                            ParseCollidersAll(m_ColliderBuffer, numFound, cellIndex, cellCenter);
                        }
                        else
                        {
                            ParseCollidersClosest(m_ColliderBuffer, numFound, cellIndex, cellCenter);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// This method attempts to perform the Physics.OverlapBoxNonAlloc and will double the size of the Collider buffer
        /// if the number of Colliders in the buffer after the call is equal to the length of the buffer.
        /// </summary>
        /// <param name="cellCenter"></param>
        /// <param name="halfCellScale"></param>
        /// <param name="rotation"></param>
        /// <returns></returns>
        int BufferResizingOverlapBoxNonAlloc(Vector3 cellCenter, Vector3 halfCellScale, Quaternion rotation)
        {
            int numFound;
            // Since we can only get a fixed number of results, requery
            // until we're sure we can hold them all (or until we hit the max size).
            while (true)
            {
                numFound = Physics.OverlapBoxNonAlloc(cellCenter, halfCellScale, m_ColliderBuffer, rotation, m_ObserveMask);
                if (numFound == m_ColliderBuffer.Length && m_ColliderBuffer.Length < m_MaxColliderBufferSize)
                {
                    m_ColliderBuffer = new Collider[Math.Min(m_MaxColliderBufferSize, m_ColliderBuffer.Length * 2)];
                    m_InitialColliderBufferSize = m_ColliderBuffer.Length;
                }
                else
                {
                    break;
                }
            }

            return numFound;
        }

        /// <summary>
        /// Parses the array of colliders found within a cell. Finds the closest gameobject to the agent root reference within the cell
        /// </summary>
        /// <param name="foundColliders">Array of the colliders found within the cell</param>
        /// <param name="numFound">Number of colliders found.</param>
        /// <param name="cellIndex">The index of the cell</param>
        /// <param name="cellCenter">The center position of the cell</param>
        void ParseCollidersClosest(Collider[] foundColliders, int numFound, int cellIndex, Vector3 cellCenter)
        {
            Profiler.BeginSample("GridSensor.ParseColliders");
            GameObject closestColliderGo = null;
            var minDistanceSquared = float.MaxValue;

            for (var i = 0; i < numFound; i++)
            {
                var currentColliderGo = foundColliders[i].gameObject;

                // Continue if the current collider go is the root reference
                if (ReferenceEquals(currentColliderGo, m_RootReference))
                    continue;

                var closestColliderPoint = foundColliders[i].ClosestPointOnBounds(cellCenter);
                var currentDistanceSquared = (closestColliderPoint - m_RootReference.transform.position).sqrMagnitude;

                // Checks if our colliders contain a detectable object
                var index = -1;
                for (var ii = 0; ii < m_DetectableObjects.Length; ii++)
                {
                    if (currentColliderGo.CompareTag(m_DetectableObjects[ii]))
                    {
                        index = ii;
                        break;
                    }
                }
                if (index > -1 && currentDistanceSquared < minDistanceSquared)
                {
                    minDistanceSquared = currentDistanceSquared;
                    closestColliderGo = currentColliderGo;
                }
            }

            if (!ReferenceEquals(closestColliderGo, null))
                LoadObjectData(closestColliderGo, cellIndex, (float)Math.Sqrt(minDistanceSquared) * m_InverseSphereRadius);
            Profiler.EndSample();
        }

        /// <summary>
        /// For each collider, calls LoadObjectData on the gameobejct
        /// </summary>
        /// <param name="foundColliders">The array of colliders</param>
        /// <param name="cellIndex">The cell index the collider is in</param>
        /// <param name="cellCenter">the center of the cell the collider is in</param>
        void ParseCollidersAll(Collider[] foundColliders, int numFound, int cellIndex, Vector3 cellCenter)
        {
            Profiler.BeginSample("GridSensor.ParseColliders");
            GameObject currentColliderGo = null;
            Vector3 closestColliderPoint = Vector3.zero;

            for (int i = 0; i < numFound; i++)
            {
                currentColliderGo = foundColliders[i].gameObject;

                // Continue if the current collider go is the root reference
                if (currentColliderGo == m_RootReference)
                    continue;

                closestColliderPoint = foundColliders[i].ClosestPointOnBounds(cellCenter);

                LoadObjectData(currentColliderGo, cellIndex,
                    Vector3.Distance(closestColliderPoint, m_RootReference.transform.position) * m_InverseSphereRadius);
            }
            Profiler.EndSample();
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
        protected virtual float[] GetObjectData(GameObject currentColliderGo, float typeIndex, float normalizedDistance)
        {
            if (m_ChannelBuffer == null)
            {
                m_ChannelBuffer = new float[m_ChannelDepth.Length];
            }
            Array.Clear(m_ChannelBuffer, 0, m_ChannelBuffer.Length);
            m_ChannelBuffer[0] = typeIndex;
            return m_ChannelBuffer;
        }

        /// <summary>
        /// Runs basic validation assertions to check that the values can be normalized
        /// </summary>
        /// <param name="channelValues">The values to be validated</param>
        /// <param name="currentColliderGo">The gameobject used for better error messages</param>
        protected virtual void ValidateValues(float[] channelValues, GameObject currentColliderGo)
        {
            for (int j = 0; j < channelValues.Length; j++)
            {
                if (channelValues[j] < 0)
                    throw new UnityAgentsException("Expected ChannelValue[" + j + "] for " + currentColliderGo.name + " to be non-negative, was " + channelValues[j]);

                if (channelValues[j] > m_ChannelDepth[j])
                    throw new UnityAgentsException("Expected ChannelValue[" + j + "]  for " + currentColliderGo.name + " to be less than ChannelDepth[" + j + "] (" + m_ChannelDepth[j] + "), was " + channelValues[j]);
            }
        }

        /// <summary>
        /// LoadObjectData - If the GameObject matches a tag, GetObjectData is called to extract the data from the GameObject
        /// then the data is transformed based on the GridDepthType of the gridsensor.
        /// Further documetation on the GridDepthType can be found below
        /// </summary>
        /// <param name="currentColliderGo">The game object that was found colliding with a certain cell</param>
        /// <param name="cellIndex">The index of the current cell</param>
        /// <param name="normalizedDistance">A float between 0 and 1 describing the ratio of
        ///            the distance currentColliderGo is compared to the edge of the gridsensor</param>
        protected virtual void LoadObjectData(GameObject currentColliderGo, int cellIndex, float normalizedDistance)
        {
            Profiler.BeginSample("GridSensor.LoadObjectData");
            var channelHotVals = new ArraySegment<float>(m_PerceptionBuffer, cellIndex * m_ObservationPerCell, m_ObservationPerCell);
            for (var i = 0; i < m_DetectableObjects.Length; i++)
            {
                for (var ii = 0; ii < channelHotVals.Count; ii++)
                {
                    m_PerceptionBuffer[channelHotVals.Offset + ii] = 0f;
                }

                if (!ReferenceEquals(currentColliderGo, null) && currentColliderGo.CompareTag(m_DetectableObjects[i]))
                {
                    // TODO: Create the array already then set the values using "out" in GetObjectData
                    // Using i+1 as the type index as "0" represents "empty"
                    var channelValues = GetObjectData(currentColliderGo, (float)i + 1, normalizedDistance);
                    ValidateValues(channelValues, currentColliderGo);

                    if (m_ShowGizmos)
                    {
                        m_CellActivity[cellIndex] = i;
                    }

                    switch (m_GridDepthType)
                    {
                        case GridDepthType.Channel:
                            {
                                // The observations are "channel based" so each grid is WxHxC where C is the number of channels
                                // This typically means that each channel value is normalized between 0 and 1
                                // If channelDepth is 1, the value is assumed normalized, else the value is normalized by the channelDepth
                                // The channels are then stored consecutively in PerceptionBuffer.
                                // NOTE: This is the only grid type that uses floating point values
                                // For example, if a cell contains the 3rd type of 5 possible on the 2nd team of 3 possible teams:
                                // channelValues = {2, 1}
                                // ObservationPerCell = channelValues.Length
                                // channelValues = {2f/5f, 1f/3f} = {.4, .33..}
                                // Array.Copy(channelValues, 0, PerceptionBuffer, cell_id*ObservationPerCell, ObservationPerCell);
                                for (int j = 0; j < channelValues.Length; j++)
                                {
                                    channelValues[j] /= m_ChannelDepth[j];
                                }

                                Array.Copy(channelValues, 0, m_PerceptionBuffer, cellIndex * m_ObservationPerCell, m_ObservationPerCell);
                                break;
                            }

                        case GridDepthType.ChannelHot:
                            {
                                // The observations are "channel hot" so each grid is WxHxD where D is the sum of all of the channel depths
                                // The opposite of the "channel based" case, the channel values are represented as one hot vector per channel and then concatenated together
                                // Thus channelDepth is assumed to be greater than 1.
                                // For example, if a cell contains the 3rd type of 5 possible on the 2nd team of 3 possible teams,
                                // channelValues = {2, 1}
                                // channelOffsets = {5, 3}
                                // ObservationPerCell = 5 + 3 = 8
                                // channelHotVals = {0, 0, 1, 0, 0, 0, 1, 0}
                                // Array.Copy(channelHotVals, 0, PerceptionBuffer, cell_id*ObservationPerCell, ObservationPerCell);
                                for (int j = 0; j < channelValues.Length; j++)
                                {
                                    if (m_ChannelDepth[j] > 1)
                                    {
                                        m_PerceptionBuffer[channelHotVals.Offset + (int)channelValues[j] + m_ChannelOffsets[j]] = 1f;
                                    }
                                    else
                                    {
                                        m_PerceptionBuffer[channelHotVals.Offset + m_ChannelOffsets[j]] = channelValues[j];
                                    }
                                }
                                break;
                            }
                        case GridDepthType.Counting:
                            {
                                // The observations are "channel count" so each grid is WxHxC where C is the number of tags
                                // This means that each value channelValues[i] is a counter of gameobject included into grid cells
                                // where i is the index of the tag in DetectableObjects
                                int countIndex = cellIndex * m_ObservationPerCell + i;
                                m_PerceptionBuffer[countIndex] = Mathf.Min(1f, m_PerceptionBuffer[countIndex] + 1f / m_ChannelDepth[i]);
                                break;
                            }
                    }

                    break;
                }
            }
            Profiler.EndSample();
        }

        /// <summary>Converts the index of the cell to the 3D point (y is zero) relative to grid center</summary>
        /// <returns>Vector3 of the position of the center of the cell relative to grid center</returns>
        /// <param name="cell">The index of the cell</param>
        Vector3 CellToPoint(int cellIndex)
        {
            float x = (cellIndex % m_GridNumSideZ - m_OffsetGridNumSide) * m_CellScaleX;
            float z = (cellIndex / m_GridNumSideZ - m_OffsetGridNumSide) * m_CellScaleZ - (m_GridNumSideZ - m_GridNumSideX);
            return new Vector3(x, 0, z);
        }

        internal Vector3 GetCellGlobalPosition(int cellIndex)
        {
            if (m_RotateWithAgent)
            {
                return m_RootReference.transform.TransformPoint(m_CellPoints[cellIndex]);
            }
            else
            {
                return m_CellPoints[cellIndex] + m_RootReference.transform.position;
            }
        }

        internal Quaternion GetGridRotation()
        {
            return m_RotateWithAgent ? m_RootReference.transform.rotation : Quaternion.identity;
        }

        /// <inheritdoc/>
        public void Update()
        {
            using (TimerStack.Instance.Scoped("GridSensor.Update"))
            {
                Perceive();
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
                for (var h = m_GridNumSideZ - 1; h >= 0; h--) // height
                {
                    for (var w = 0; w < m_GridNumSideX; w++) // width
                    {
                        for (var d = 0; d < m_ObservationPerCell; d++) // depth
                        {
                            writer[h, w, d] = m_PerceptionBuffer[index];
                            index++;
                        }
                    }
                }
                return index;
            }
        }
    }
}
