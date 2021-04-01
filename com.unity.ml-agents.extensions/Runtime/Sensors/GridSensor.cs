using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.MLAgents.Sensors;
using UnityEngine.Profiling;

[assembly: InternalsVisibleTo("Unity.ML-Agents.Extensions.EditorTests")]
namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Enum describing what kind of depth type the data should be organized as
    /// </summary>
    public enum GridDepthType { Channel, ChannelHot, Counting };

    /// <summary>
    /// Grid-based sensor.
    /// </summary>
    public class GridSensor : ISensor, IBuiltInSensor
    {
        /// <summary>
        /// Name of this grid sensor.
        /// </summary>
        string Name;

        //
        // Main Parameters
        //

        /// <summary>
        /// The scale of each grid cell.
        /// </summary>
        Vector3 CellScale;

        /// <summary>
        /// The number of grid on each side.
        /// </summary>
        Vector3Int GridNumSide;

        /// <summary>
        /// Rotate the grid based on the direction the agent is facing.
        /// </summary>
        bool RotateWithAgent;

        /// <summary>
        /// Array holding the depth of each channel.
        /// </summary>
        int[] ChannelDepth;

        /// <summary>
        /// List of tags that are detected.
        /// </summary>
        string[] DetectableObjects;

        /// <summary>
        /// The layer mask.
        /// </summary>
        LayerMask ObserveMask;

        /// <summary>
        /// The data layout that the grid should output.
        /// </summary>
        GridDepthType gridDepthType = GridDepthType.Channel;

        /// <summary>
        /// The reference of the root of the agent. This is used to disambiguate objects with the same tag as the agent. Defaults to current GameObject.
        /// </summary>
        GameObject rootReference;

        int MaxColliderBufferSize;

        int InitialColliderBufferSize;
        Collider[] m_ColliderBuffer;

        float[] m_ChannelBuffer;

        //
        // Hidden Parameters
        //

        /// <summary>
        /// The total number of observations per cell of the grid. Its equivalent to the "channel" on the outgoing tensor.
        /// </summary>
        int ObservationPerCell;

        /// <summary>
        /// The offsets used to specify where within a cell's allotted data, certain observations will be inserted.
        /// </summary>
        int[] ChannelOffsets;

        /// <summary>
        /// The main storage of perceptual information.
        /// </summary>
        internal float[] m_PerceptionBuffer;

        /// <summary>
        ///  The default value of the perceptionBuffer when using the ChannelHot DepthType. Used to reset the array/
        /// </summary>
        float[] m_ChannelHotDefaultPerceptionBuffer;

        /// <summary>
        /// Array of Colors needed in order to load the values of the perception buffer to a texture.
        /// </summary>
        Color[] m_PerceptionColors;

        /// <summary>
        /// Texture where the colors are written to so that they can be compressed in PNG format.
        /// </summary>
        Texture2D m_perceptionTexture2D;

        //
        // Utility Constants Calculated on Init
        //

        /// <summary>
        /// Radius of grid, used for normalizing the distance.
        /// </summary>
        float InverseSphereRadius;

        /// <summary>
        /// Total Number of cells (width*height)
        /// </summary>
        int NumCells;

        /// <summary>
        /// Offset used for calculating CellToPoint
        /// </summary>
        float OffsetGridNumSide = 7.5f; //  (gridNumSideZ - 1) / 2;

        /// <summary>
        /// Cached ObservationSpec
        /// </summary>
        ObservationSpec m_ObservationSpec;

        //
        // Debug Parameters
        //

        SensorCompressionType m_CompressionType = SensorCompressionType.PNG;

        /// <summary>
        /// Array of colors displaying the DebugColors for each cell in OnDrawGizmos. Only updated if ShowGizmos.
        /// </summary>
        int[] m_CellActivity;

        /// <summary>
        /// Array of global positions where each position is the center of a cell.
        /// </summary>
        Vector3[] m_GizmoCellPosition;

        /// <summary>
        /// Array of local positions where each position is the center of a cell.
        /// </summary>
        Vector3[] CellPoints;

        public GridSensor(
            string name,
            Vector3 cellScale,
            Vector3Int gridNumSide,
            bool rotateWithAgent,
            int[] channelDepth,
            string[] detectableObjects,
            LayerMask observeMask,
            GridDepthType depthType,
            GameObject root,
            SensorCompressionType compression,
            int maxColliderBufferSize,
            int initialColliderBufferSize
        )
        {
            Name = name;
            CellScale = cellScale;
            GridNumSide = gridNumSide;
            if (GridNumSide.y != 1)
            {
                throw new UnityAgentsException("GridSensor only supports 2D grids.");
            }
            RotateWithAgent = rotateWithAgent;
            ChannelDepth = channelDepth;
            DetectableObjects = detectableObjects;
            ObserveMask = observeMask;
            gridDepthType = depthType;
            rootReference = root;
            CompressionType = compression;
            MaxColliderBufferSize = maxColliderBufferSize;
            InitialColliderBufferSize = initialColliderBufferSize;

            if (gridDepthType == GridDepthType.Counting && DetectableObjects.Length != ChannelDepth.Length)
            {
                throw new UnityAgentsException("The channels of a CountingGridSensor is equal to the number of detectableObjects");
            }

            InitGridParameters();
            InitDepthType();
            InitCellPoints();
            ResetPerceptionBuffer();

            m_ObservationSpec = ObservationSpec.Visual(GridNumSide.x, GridNumSide.z, ObservationPerCell);
            m_perceptionTexture2D = new Texture2D(GridNumSide.x, GridNumSide.z, TextureFormat.RGB24, false);
            m_ColliderBuffer = new Collider[Math.Min(MaxColliderBufferSize, InitialColliderBufferSize)];
        }

        public SensorCompressionType CompressionType
        {
            get { return m_CompressionType; }
            set { m_CompressionType = value; }
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
            NumCells = GridNumSide.x * GridNumSide.z;
            float sphereRadiusX = (CellScale.x * GridNumSide.x) / Mathf.Sqrt(2);
            float sphereRadiusZ = (CellScale.z * GridNumSide.z) / Mathf.Sqrt(2);
            InverseSphereRadius = 1.0f / Mathf.Max(sphereRadiusX, sphereRadiusZ);
            OffsetGridNumSide = (GridNumSide.z - 1f) / 2f;
        }

        /// <summary>
        /// Initializes the constant parameters that are based on the Grid Depth Type
        /// Sets the ObservationPerCell and the ChannelOffsets properties
        /// </summary>
        void InitDepthType()
        {
            if (gridDepthType == GridDepthType.ChannelHot)
            {
                ObservationPerCell = ChannelDepth.Sum();

                ChannelOffsets = new int[ChannelDepth.Length];
                for (int i = 1; i < ChannelDepth.Length; i++)
                {
                    ChannelOffsets[i] = ChannelOffsets[i - 1] + ChannelDepth[i - 1];
                }

                m_ChannelHotDefaultPerceptionBuffer = new float[ObservationPerCell];
                for (int i = 0; i < ChannelDepth.Length; i++)
                {
                    if (ChannelDepth[i] > 1)
                    {
                        m_ChannelHotDefaultPerceptionBuffer[ChannelOffsets[i]] = 1;
                    }
                }
            }
            else
            {
                ObservationPerCell = ChannelDepth.Length;
            }

            // The maximum number of channels in the final output must be less than 255 * 3 because the "number of PNG images" to generate must fit in one byte
            Assert.IsTrue(ObservationPerCell < (255 * 3), "The maximum number of channels per cell must be less than 255 * 3");
        }

        /// <summary>
        /// Initializes the location of the CellPoints property
        /// </summary>
        void InitCellPoints()
        {
            CellPoints = new Vector3[NumCells];

            for (int i = 0; i < NumCells; i++)
            {
                CellPoints[i] = CellToLocalPosition(i);
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
                if (gridDepthType == GridDepthType.ChannelHot)
                {
                    // Copy the default value to the array
                    for (int i = 0; i < NumCells; i++)
                    {
                        Array.Copy(m_ChannelHotDefaultPerceptionBuffer, 0, m_PerceptionBuffer, i * ObservationPerCell, ObservationPerCell);
                    }
                }
                else
                {
                    Array.Clear(m_PerceptionBuffer, 0, m_PerceptionBuffer.Length);
                }
            }
            else
            {
                m_PerceptionBuffer = new float[ObservationPerCell * NumCells];
                m_ColliderBuffer = new Collider[Math.Min(MaxColliderBufferSize, InitialColliderBufferSize)];
                m_ChannelBuffer = new float[ChannelDepth.Length];
                m_PerceptionColors = new Color[NumCells];
                m_GizmoCellPosition = new Vector3[NumCells];
            }
        }

        public void ResetGizmoBuffer()
        {
            // Ensure to init arrays if not yet assigned (for editor)
            if (m_CellActivity == null)
                m_CellActivity = new int[NumCells];

            // Assign the default color to the cell activities
            for (int i = 0; i < NumCells; i++)
            {
                m_CellActivity[i] = -1;
            }
        }


        /// <inheritdoc/>
        public string GetName()
        {
            return Name;
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
                var numImages = (ObservationPerCell + 2) / 3;
                for (int i = 0; i < numImages; i++)
                {
                    var channelIndex = 3 * i;
                    ChannelsToTexture(channelIndex, Math.Min(3, ObservationPerCell - channelIndex));
                    allBytes.AddRange(m_perceptionTexture2D.EncodeToPNG());
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
            for (int i = 0; i < NumCells; i++)
            {
                for (int j = 0; j < numChannelsToAdd; j++)
                {
                    m_PerceptionColors[i][j] = m_PerceptionBuffer[i * ObservationPerCell + channelIndex + j];
                }
            }
            m_perceptionTexture2D.SetPixels(m_PerceptionColors);
        }

        /// <summary>
        /// Perceive - Clears the buffers, calls overlap box on the actual cell (the actual perception part)
        /// for all found colliders, LoadObjectData is called
        /// </summary>
        internal void Perceive()
        {
            if (m_ColliderBuffer == null)
            {
                return;
            }

            ResetPerceptionBuffer();
            using (TimerStack.Instance.Scoped("GridSensor.Perceive"))
            {
                var halfCellScale = new Vector3(CellScale.x / 2f, CellScale.y, CellScale.z / 2f);

                for (var cellIndex = 0; cellIndex < NumCells; cellIndex++)
                {
                    var cellCenter = GetCellGlobalPosition(cellIndex);
                    var numFound = BufferResizingOverlapBoxNonAlloc(cellCenter, halfCellScale, GetGridRotation());

                    if (numFound > 0)
                    {
                        if (gridDepthType == GridDepthType.Counting)
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
                numFound = Physics.OverlapBoxNonAlloc(cellCenter, halfCellScale, m_ColliderBuffer, rotation, ObserveMask);
                if (numFound == m_ColliderBuffer.Length && m_ColliderBuffer.Length < MaxColliderBufferSize)
                {
                    m_ColliderBuffer = new Collider[Math.Min(MaxColliderBufferSize, m_ColliderBuffer.Length * 2)];
                    InitialColliderBufferSize = m_ColliderBuffer.Length;
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
                if (ReferenceEquals(currentColliderGo, rootReference))
                    continue;

                var closestColliderPoint = foundColliders[i].ClosestPointOnBounds(cellCenter);
                var currentDistanceSquared = (closestColliderPoint - rootReference.transform.position).sqrMagnitude;

                // Checks if our colliders contain a detectable object
                var index = -1;
                for (var ii = 0; ii < DetectableObjects.Length; ii++)
                {
                    if (currentColliderGo.CompareTag(DetectableObjects[ii]))
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
            {
                LoadObjectData(closestColliderGo, cellIndex, (float)Math.Sqrt(minDistanceSquared) * InverseSphereRadius);
            }
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
                if (currentColliderGo == rootReference)
                    continue;

                closestColliderPoint = foundColliders[i].ClosestPointOnBounds(cellCenter);

                LoadObjectData(currentColliderGo, cellIndex,
                    Vector3.Distance(closestColliderPoint, rootReference.transform.position) * InverseSphereRadius);
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

                if (channelValues[j] > ChannelDepth[j])
                    throw new UnityAgentsException("Expected ChannelValue[" + j + "]  for " + currentColliderGo.name + " to be less than ChannelDepth[" + j + "] (" + ChannelDepth[j] + "), was " + channelValues[j]);
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
            var channelHotVals = new ArraySegment<float>(m_PerceptionBuffer, cellIndex * ObservationPerCell, ObservationPerCell);
            for (var i = 0; i < DetectableObjects.Length; i++)
            {
                if (gridDepthType != GridDepthType.Counting)
                {
                    for (var ii = 0; ii < channelHotVals.Count; ii++)
                    {
                        m_PerceptionBuffer[channelHotVals.Offset + ii] = 0f;
                    }
                }

                if (!ReferenceEquals(currentColliderGo, null) && currentColliderGo.CompareTag(DetectableObjects[i]))
                {
                    // TODO: Create the array already then set the values using "out" in GetObjectData
                    // Using i+1 as the type index as "0" represents "empty"
                    var channelValues = GetObjectData(currentColliderGo, (float)i + 1, normalizedDistance);
                    ValidateValues(channelValues, currentColliderGo);

                    switch (gridDepthType)
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
                                    channelValues[j] /= ChannelDepth[j];
                                }

                                Array.Copy(channelValues, 0, m_PerceptionBuffer, cellIndex * ObservationPerCell, ObservationPerCell);
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
                                    if (ChannelDepth[j] > 1)
                                    {
                                        m_PerceptionBuffer[channelHotVals.Offset + (int)channelValues[j] + ChannelOffsets[j]] = 1f;
                                    }
                                    else
                                    {
                                        m_PerceptionBuffer[channelHotVals.Offset + ChannelOffsets[j]] = channelValues[j];
                                    }
                                }
                                break;
                            }
                        case GridDepthType.Counting:
                            {
                                // The observations are "channel count" so each grid is WxHxC where C is the number of tags
                                // This means that each value channelValues[i] is a counter of gameobject included into grid cells
                                // where i is the index of the tag in DetectableObjects
                                int countIndex = cellIndex * ObservationPerCell + i;
                                m_PerceptionBuffer[countIndex] = Mathf.Min(1f, m_PerceptionBuffer[countIndex] + 1f / ChannelDepth[i]);
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
        Vector3 CellToLocalPosition(int cellIndex)
        {
            float x = (cellIndex % GridNumSide.z - OffsetGridNumSide) * CellScale.x;
            float z = (cellIndex / GridNumSide.z - OffsetGridNumSide) * CellScale.z - (GridNumSide.z - GridNumSide.x);
            return new Vector3(x, 0, z);
        }

        internal Vector3 GetCellGlobalPosition(int cellIndex)
        {
            if (RotateWithAgent)
            {
                return rootReference.transform.TransformPoint(CellPoints[cellIndex]);
            }
            else
            {
                return CellPoints[cellIndex] + rootReference.transform.position;
            }
        }

        internal Quaternion GetGridRotation()
        {
            return RotateWithAgent ? rootReference.transform.rotation : Quaternion.identity;
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
                for (var h = GridNumSide.z - 1; h >= 0; h--)
                {
                    for (var w = 0; w < GridNumSide.x; w++)
                    {
                        for (var d = 0; d < ObservationPerCell; d++)
                        {
                            writer[h, w, d] = m_PerceptionBuffer[index];
                            index++;
                        }
                    }
                }
                return index;
            }
        }

        internal int[] PerceiveGizmoColor()
        {
            ResetGizmoBuffer();
            var halfCellScale = new Vector3(CellScale.x / 2f, CellScale.y, CellScale.z / 2f);

            for (var cellIndex = 0; cellIndex < NumCells; cellIndex++)
            {
                var cellCenter = GetCellGlobalPosition(cellIndex);
                var numFound = BufferResizingOverlapBoxNonAlloc(cellCenter, halfCellScale, GetGridRotation());

                var minDistanceSquared = float.MaxValue;
                var tagIndex = -1;

                for (var i = 0; i < numFound; i++)
                {
                    var currentColliderGo = m_ColliderBuffer[i].gameObject;
                    if (ReferenceEquals(currentColliderGo, rootReference))
                        continue;

                    var closestColliderPoint = m_ColliderBuffer[i].ClosestPointOnBounds(cellCenter);
                    var currentDistanceSquared = (closestColliderPoint - rootReference.transform.position).sqrMagnitude;

                    // Checks if our colliders contain a detectable object
                    var index = -1;
                    for (var ii = 0; ii < DetectableObjects.Length; ii++)
                    {
                        if (currentColliderGo.CompareTag(DetectableObjects[ii]))
                        {
                            index = ii;
                            break;
                        }
                    }
                    if (index > -1 && currentDistanceSquared < minDistanceSquared)
                    {
                        minDistanceSquared = currentDistanceSquared;
                        tagIndex = index;
                    }
                }
                CellActivity[cellIndex] = tagIndex;
            }
            return CellActivity;
        }

        internal Vector3[] GetGizmoPositions()
        {
            for (var i = 0; i < NumCells; i++)
            {
                m_GizmoCellPosition[i] = GetCellGlobalPosition(i);
            }
            return m_GizmoCellPosition;
        }
    }
}
