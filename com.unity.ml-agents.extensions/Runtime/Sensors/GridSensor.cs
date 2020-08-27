using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Grid-based sensor.
    /// </summary>
    public class GridSensor : SensorComponent, ISensor
    {
        /// <summary>
        /// Name of this grid sensor.
        /// </summary>
        public string Name;

        //
        // Main Parameters
        //

        /// <summary>
        /// The width of each grid cell.
        /// </summary>
        [Header("Grid Sensor Settings")]
        [Tooltip("The width of each grid cell")]
        [Range(0.05f, 1000f)]
        public float CellScaleX = 1f;

        /// <summary>
        /// The depth of each grid cell.
        /// </summary>
        [Tooltip("The depth of each grid cell")]
        [Range(0.05f, 1000f)]
        public float CellScaleZ = 1f;

        /// <summary>
        /// The width of the grid .
        /// </summary>
        [Tooltip("The width of the grid")]
        [Range(2, 2000)]
        public int GridNumSideX = 16;

        /// <summary>
        /// The depth of the grid .
        /// </summary>
        [Tooltip("The depth of the grid")]
        [Range(2, 2000)]
        public int GridNumSideZ = 16;

        /// <summary>
        /// The height of each grid cell. Changes how much of the vertical axis is observed by a cell.
        /// </summary>
        [Tooltip("The height of each grid cell. Changes how much of the vertical axis is observed by a cell")]
        [Range(0.01f, 1000f)]
        public float CellScaleY = 0.01f;

        /// <summary>
        /// Rotate the grid based on the direction the agent is facing.
        /// </summary>
        [Tooltip("Rotate the grid based on the direction the agent is facing")]
        public bool RotateToAgent;

        /// <summary>
        /// Array holding the depth of each channel.
        /// </summary>
        [Tooltip("Array holding the depth of each channel")]
        public int[] ChannelDepth;

        /// <summary>
        /// List of tags that are detected.
        /// </summary>
        [Tooltip("List of tags that are detected")]
        public string[] DetectableObjects;

        /// <summary>
        /// The layer mask.
        /// </summary>
        [Tooltip("The layer mask")]
        public LayerMask ObserveMask;

        /// <summary>
        /// Enum describing what kind of depth type the data should be organized as
        /// </summary>
        public enum GridDepthType { Channel, ChannelHot };

        /// <summary>
        /// The data layout that the grid should output.
        /// </summary>
        [Tooltip("The data layout that the grid should output")]
        public GridDepthType gridDepthType = GridDepthType.Channel;

        /// <summary>
        /// The reference of the root of the agent. This is used to disambiguate objects with the same tag as the agent. Defaults to current GameObject.
        /// </summary>
        [Tooltip("The reference of the root of the agent. This is used to disambiguate objects with the same tag as the agent. Defaults to current GameObject")]
        public GameObject rootReference;

        //
        // Hidden Parameters
        //

        /// <summary>
        /// The total number of observations per cell of the grid. Its equivalent to the "channel" on the outgoing tensor.
        /// </summary>
        [HideInInspector]
        public int ObservationPerCell;

        /// <summary>
        /// The total number of observations that this GridSensor provides. It is the length of m_PerceptionBuffer.
        /// </summary>
        [HideInInspector]
        public int NumberOfObservations;

        /// <summary>
        /// The offsets used to specify where within a cell's allotted data, certain observations will be inserted.
        /// </summary>
        [HideInInspector]
        public int[] ChannelOffsets;

        /// <summary>
        /// The main storage of perceptual information.
        /// </summary>
        protected float[] m_PerceptionBuffer;

        /// <summary>
        ///  The default value of the perceptionBuffer when using the ChannelHot DepthType. Used to reset the array/
        /// </summary>
        protected float[] m_ChannelHotDefaultPerceptionBuffer;

        /// <summary>
        /// Array of Colors needed in order to load the values of the perception buffer to a texture.
        /// </summary>
        protected Color[] m_PerceptionColors;

        /// <summary>
        /// Texture where the colors are written to so that they can be compressed in PNG format.
        /// </summary>
        protected Texture2D m_perceptionTexture2D;

        //
        // Utility Constants Calculated on Init
        //

        /// <summary>
        /// Number of PNG formated images that are sent to python during training.
        /// </summary>
        private int NumImages;

        /// <summary>
        /// Number of relevant channels on the last image that is sent/
        /// </summary>
        private int NumChannelsOnLastImage;

        /// <summary>
        /// Radius of grid, used for normalizing the distance.
        /// </summary>
        protected float SphereRadius;

        /// <summary>
        /// Total Number of cells (width*height)
        /// </summary>
        private int NumCells;

        /// <summary>
        /// Difference between GridNumSideZ and gridNumSideX.
        /// </summary>
        protected int DiffNumSideZX = 0;

        /// <summary>
        /// Offset used for calculating CellToPoint
        /// </summary>
        protected float OffsetGridNumSide = 7.5f; //  (gridNumSideZ - 1) / 2;

        /// <summary>
        /// Half of the grid in the X direction
        /// </summary>
        private float HalfOfGridX;

        /// <summary>
        /// Half of the grid in the z direction
        /// </summary>
        private float HalfOfGridZ;

        /// <summary>
        /// Used in the PointToCell method to scale the x value to land in the calculated cell.
        /// </summary>
        private float PointToCellScalingX;

        /// <summary>
        /// Used in the PointToCell method to scale the y value to land in the calculated cell.
        /// </summary>
        private float PointToCellScalingZ;

        /// <summary>
        /// Bool if initialized or not.
        /// </summary>
        protected bool Initialized = false;

        /// <summary>
        /// Array holding the dimensions of the resulting tensor
        /// </summary>
        private int[] m_Shape;

        //
        // Debug Parameters
        //

        /// <summary>
        /// Array of Colors used for the grid gizmos.
        /// </summary>
        [Header("Debug Options")]
        [Tooltip("Array of Colors used for the grid gizmos")]
        public Color[] DebugColors;

        /// <summary>
        /// The height of the gizmos grid.
        /// </summary>
        [Tooltip("The height of the gizmos grid")]
        public float GizmoYOffset = 0f;

        /// <summary>
        /// Whether to show gizmos or not.
        /// </summary>
        [Tooltip("Whether to show gizmos or not")]
        public bool ShowGizmos = false;

        /// <summary>
        /// Array of colors displaying the DebugColors for each cell in OnDrawGizmos. Only updated if ShowGizmos.
        /// </summary>
        protected Color[] CellActivity;

        /// <summary>
        /// Array of positions where each position is the center of a cell.
        /// </summary>
        private Vector3[] CellPoints;

        /// <summary>
        /// List representing the multiple compressed images of all of the grids
        /// </summary>
        private List<byte[]> compressedImgs;

        /// <summary>
        /// List representing the sizes of the multiple images so they can be properly reconstructed on the python side
        /// </summary>
        private List<byte[]> byteSizesBytesList;

        private Color DebugDefaultColor = new Color(1f, 1f, 1f, 0.25f);

        /// <inheritdoc/>
        public override ISensor CreateSensor()
        {
            return this;
        }

        /// <summary>
        /// Sets the parameters of the grid sensor
        /// </summary>
        /// <param name="detectableObjects">array of strings representing the tags to be detected by the sensor</param>
        /// <param name="channelDepth">array of ints representing the depth of each channel</param>
        /// <param name="gridDepthType">enum representing the GridDepthType of the sensor</param>
        /// <param name="cellScaleX">float representing the X scaling of each cell</param>
        /// <param name="cellScaleZ">float representing the Z scaling of each cell</param>
        /// <param name="gridWidth">int representing the number of cells in the X direction. Width of the Grid</param>
        /// <param name="gridHeight">int representing the number of cells in the Z direction. Height of the Grid</param>
        /// <param name="observeMaskInt">int representing the layer mask to observe</param>
        /// <param name="rotateToAgent">bool if true then the grid is rotated to the rotation of the transform the rootReference</param>
        /// <param name="debugColors">array of colors corresponding the the tags in the detectableObjects array</param>
        public virtual void SetParameters(string[] detectableObjects, int[] channelDepth, GridDepthType gridDepthType,
            float cellScaleX, float cellScaleZ, int gridWidth, int gridHeight, int observeMaskInt, bool rotateToAgent, Color[] debugColors)
        {
            this.ObserveMask = observeMaskInt;
            this.DetectableObjects = detectableObjects;
            this.ChannelDepth = channelDepth;
            this.gridDepthType = gridDepthType;
            this.CellScaleX = cellScaleX;
            this.CellScaleZ = cellScaleZ;
            this.GridNumSideX = gridWidth;
            this.GridNumSideZ = gridHeight;
            this.RotateToAgent = rotateToAgent;
            this.DiffNumSideZX = (GridNumSideZ - GridNumSideX);
            this.OffsetGridNumSide = (GridNumSideZ - 1f) / 2f;
            this.DebugColors = debugColors;
        }

        /// <summary>
        /// Initializes the constant parameters used within the perceive method call
        /// </summary>
        public void InitGridParameters()
        {
            NumCells = GridNumSideX * GridNumSideZ;
            float sphereRadiusX = (CellScaleX * GridNumSideX) / Mathf.Sqrt(2);
            float sphereRadiusZ = (CellScaleZ * GridNumSideZ) / Mathf.Sqrt(2);
            SphereRadius = Mathf.Max(sphereRadiusX, sphereRadiusZ);
            ChannelOffsets = new int[ChannelDepth.Length];
            DiffNumSideZX = (GridNumSideZ - GridNumSideX);
            OffsetGridNumSide = (GridNumSideZ - 1f) / 2f;
            HalfOfGridX = CellScaleX * GridNumSideX / 2;
            HalfOfGridZ = CellScaleZ * GridNumSideZ / 2;

            PointToCellScalingX = GridNumSideX / (CellScaleX * GridNumSideX);
            PointToCellScalingZ = GridNumSideZ / (CellScaleZ * GridNumSideZ);
        }

        /// <summary>
        /// Initializes the constant parameters that are based on the Grid Depth Type
        /// Sets the ObservationPerCell and the ChannelOffsets properties
        /// </summary>
        public virtual void InitDepthType()
        {
            switch (gridDepthType)
            {
                case GridDepthType.Channel:
                    ObservationPerCell = ChannelDepth.Length;
                    break;

                case GridDepthType.ChannelHot:
                    ObservationPerCell = 0;
                    ChannelOffsets[ChannelOffsets.Length - 1] = 0;

                    for (int i = 1; i < ChannelDepth.Length; i++)
                    {
                        ChannelOffsets[i] = ChannelOffsets[i - 1] + ChannelDepth[i - 1];
                    }

                    for (int i = 0; i < ChannelDepth.Length; i++)
                    {
                        ObservationPerCell += ChannelDepth[i];
                    }
                    break;
            }

            // The maximum number of channels in the final output must be less than 255 * 3 because the "number of PNG images" to generate must fit in one byte
            Assert.IsTrue(ObservationPerCell < (255 * 3), "The maximum number of channels per cell must be less than 255 * 3");
        }

        /// <summary>
        /// Initializes the location of the CellPoints property
        /// </summary>
        private void InitCellPoints()
        {
            CellPoints = new Vector3[NumCells];

            for (int i = 0; i < NumCells; i++)
            {
                CellPoints[i] = CellToPoint(i, false);
            }
        }

        /// <summary>
        /// Initializes the m_ChannelHotDefaultPerceptionBuffer with default data in the case that the grid depth type is ChannelHot
        /// </summary>
        public virtual void InitChannelHotDefaultPerceptionBuffer()
        {
            m_ChannelHotDefaultPerceptionBuffer = new float[ObservationPerCell];
            for (int i = 0; i < ChannelDepth.Length; i++)
            {
                if (ChannelDepth[i] > 1)
                {
                    m_ChannelHotDefaultPerceptionBuffer[ChannelOffsets[i]] = 1;
                }
            }
        }

        /// <summary>
        /// Initializes the m_PerceptionBuffer as the main data storage property
        /// Calculates the NumImages and NumChannelsOnLastImage that are used for serializing m_PerceptionBuffer
        /// </summary>
        public void InitPerceptionBuffer()
        {
            if (Application.isPlaying)
                Initialized = true;

            NumberOfObservations = ObservationPerCell * NumCells;
            m_PerceptionBuffer = new float[NumberOfObservations];
            if (gridDepthType == GridDepthType.ChannelHot)
            {
                InitChannelHotDefaultPerceptionBuffer();
            }

            m_PerceptionColors = new Color[NumCells];

            NumImages = ObservationPerCell / 3;
            NumChannelsOnLastImage = ObservationPerCell % 3;
            if (NumChannelsOnLastImage == 0)
                NumChannelsOnLastImage = 3;
            else
                NumImages += 1;

            CellActivity = new Color[NumCells];
        }

        /// <summary>
        /// Calls the initialization methods. Creates the data storing properties used to send the data
        /// Establishes
        /// </summary>
        public virtual void Start()
        {
            InitGridParameters();
            InitDepthType();
            InitCellPoints();
            InitPerceptionBuffer();
            // Default root reference to current game object
            if (rootReference == null)
                rootReference = gameObject;
            m_Shape = new[] { GridNumSideX, GridNumSideZ, ObservationPerCell };

            compressedImgs = new List<byte[]>();
            byteSizesBytesList = new List<byte[]>();

            m_perceptionTexture2D = new Texture2D(GridNumSideX, GridNumSideZ, TextureFormat.RGB24, false);
        }

        /// <summary>
        /// Clears the perception buffer before loading in new data. If the gridDepthType is ChannelHot, then it initializes the
        /// Reset() also reinits the cell activity array (for debug)
        /// </summary>
        public void Reset()
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
                m_PerceptionBuffer = new float[NumberOfObservations];
            }

            if (ShowGizmos)
            {
                // Ensure to init arrays if not yet assigned (for editor)
                if (CellActivity == null)
                    CellActivity = new Color[NumCells];

                // Assign the default color to the cell activities
                for (int i = 0; i < NumCells; i++)
                {
                    CellActivity[i] = DebugDefaultColor;
                }
            }
        }

        /// <summary>Gets the shape of the grid observation</summary>
        /// <returns>integer array shape of the grid observation</returns>
        public int[] GetFloatObservationShape()
        {
            m_Shape = new[] { GridNumSideX, GridNumSideZ, ObservationPerCell };
            return m_Shape;
        }

        /// <inheritdoc/>
        public string GetName()
        {
            return Name;
        }

        /// <inheritdoc/>
        public virtual SensorCompressionType GetCompressionType()
        {
            return SensorCompressionType.PNG;
        }

        /// <summary>
        /// GetCompressedObservation - Calls Perceive then puts the data stored on the perception buffer
        /// onto the m_perceptionTexture2D to be converted to a byte array and returned
        /// </summary>
        /// <returns>byte[] containing the compressed observation of the grid observation</returns>
        public byte[] GetCompressedObservation()
        {
            // Timer stack is in accessable due to its protection level
            //using (TimerStack.Instance.Scoped("GridSensor.GetCompressedObservation"))
            {
                Perceive(); // Fill the perception buffer with observed data

                var allBytes = new List<byte>();

                for (int i = 0; i < NumImages - 1; i++)
                {
                    ChannelsToTexture(3 * i, 3);
                    allBytes.AddRange(m_perceptionTexture2D.EncodeToPNG());
                }

                ChannelsToTexture(3 * (NumImages - 1), NumChannelsOnLastImage);
                allBytes.AddRange(m_perceptionTexture2D.EncodeToPNG());

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
        protected void ChannelsToTexture(int channelIndex, int numChannelsToAdd)
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
        /// at the end, Perceive returns the float array of the perceptions
        /// </summary>
        /// <returns>A float[] containing all of the information collected from the gridsensor</returns>
        public float[] Perceive()
        {
            Reset();
            // TODO: make these part of the class
            Collider[] foundColliders = null;
            Vector3 cellCenter = Vector3.zero;

            Vector3 halfCellScale = new Vector3(CellScaleX / 2f, CellScaleY, CellScaleZ / 2f);

            for (int cellIndex = 0; cellIndex < NumCells; cellIndex++)
            {
                if (RotateToAgent)
                {
                    cellCenter = transform.TransformPoint(CellPoints[cellIndex]);
                    foundColliders = Physics.OverlapBox(cellCenter, halfCellScale, transform.rotation, ObserveMask);
                }
                else
                {
                    cellCenter = transform.position + CellPoints[cellIndex];
                    foundColliders = Physics.OverlapBox(cellCenter, halfCellScale, Quaternion.identity, ObserveMask);
                }

                if (foundColliders != null && foundColliders.Length > 0)
                {
                    ParseColliders(foundColliders, cellIndex, cellCenter);
                }
            }

            return m_PerceptionBuffer;
        }

        /// <summary>
        /// Parses the array of colliders found within a cell. Finds the closest gameobject to the agent root reference within the cell
        /// </summary>
        /// <param name="foundColliders">Array of the colliders found within the cell</param>
        /// <param name="cellIndex">The index of the cell</param>
        /// <param name="cellCenter">The center position of the cell</param>
        protected virtual void ParseColliders(Collider[] foundColliders, int cellIndex, Vector3 cellCenter)
        {
            GameObject currentColliderGo = null;
            GameObject closestColliderGo = null;
            Vector3 closestColliderPoint = Vector3.zero;
            float distance = float.MaxValue;
            float currentDistance = 0f;

            for (int i = 0; i < foundColliders.Length; i++)
            {
                currentColliderGo = foundColliders[i].gameObject;

                // Continue if the current collider go is the root reference
                if (currentColliderGo == rootReference)
                    continue;

                closestColliderPoint = foundColliders[i].ClosestPointOnBounds(cellCenter);
                currentDistance = Vector3.Distance(closestColliderPoint, rootReference.transform.position);

                // Checks if our colliders contain a detectable object
                if ((Array.IndexOf(DetectableObjects, currentColliderGo.tag) > -1) && (currentDistance < distance))
                {
                    distance = currentDistance;
                    closestColliderGo = currentColliderGo;
                }
            }

            if (closestColliderGo != null)
                LoadObjectData(closestColliderGo, cellIndex, distance / SphereRadius);
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
            float[] channelValues = new float[ChannelDepth.Length];
            channelValues[0] = typeIndex;
            return channelValues;
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
        /// <param name="normalized_distance">A float between 0 and 1 describing the ratio of
        ///            the distance currentColliderGo is compared to the edge of the gridsensor</param>
        protected virtual void LoadObjectData(GameObject currentColliderGo, int cellIndex, float normalized_distance)
        {
            for (int i = 0; i < DetectableObjects.Length; i++)
            {
                if (currentColliderGo != null && currentColliderGo.CompareTag(DetectableObjects[i]))
                {
                    // TODO: Create the array already then set the values using "out" in GetObjectData
                    // Using i+1 as the type index as "0" represents "empty"
                    float[] channelValues = GetObjectData(currentColliderGo, (float)i + 1, normalized_distance);

                    ValidateValues(channelValues, currentColliderGo);

                    if (ShowGizmos)
                    {
                        Color debugRayColor = Color.white;
                        if (DebugColors.Length > 0)
                        {
                            debugRayColor = DebugColors[i];
                        }
                        CellActivity[cellIndex] = new Color(debugRayColor.r, debugRayColor.g, debugRayColor.b, .5f);
                    }

                    switch (gridDepthType)
                    {
                        case GridDepthType.Channel:
                            /// <remarks>
                            /// The observations are "channel based" so each grid is WxHxC where C is the number of channels
                            /// This typically means that each channel value is normalized between 0 and 1
                            /// If channelDepth is 1, the value is assumed normalized, else the value is normalized by the channelDepth
                            /// The channels are then stored consecutively in PerceptionBuffer.
                            /// NOTE: This is the only grid type that uses floating point values
                            /// For example, if a cell contains the 3rd type of 5 possible on the 2nd team of 3 possible teams:
                            /// channelValues = {2, 1}
                            /// ObservationPerCell = channelValues.Length
                            /// channelValues = {2f/5f, 1f/3f} = {.4, .33..}
                            /// Array.Copy(channelValues, 0, PerceptionBuffer, cell_id*ObservationPerCell, ObservationPerCell);
                            /// </remarks>
                            for (int j = 0; j < channelValues.Length; j++)
                            {
                                channelValues[j] /= ChannelDepth[j];
                            }
                            Array.Copy(channelValues, 0, m_PerceptionBuffer, cellIndex * ObservationPerCell, ObservationPerCell);
                            break;

                        case GridDepthType.ChannelHot:
                            /// <remarks>
                            /// The observations are "channel hot" so each grid is WxHxD where D is the sum of all of the channel depths
                            /// The opposite of the "channel based" case, the channel values are represented as one hot vector per channel and then concatenated together
                            /// Thus channelDepth is assumed to be greater than 1.
                            /// For example, if a cell contains the 3rd type of 5 possible on the 2nd team of 3 possible teams,
                            /// channelValues = {2, 1}
                            /// channelOffsets = {5, 3}
                            /// ObservationPerCell = 5 + 3 = 8
                            /// channelHotVals = {0, 0, 1, 0, 0, 0, 1, 0}
                            /// Array.Copy(channelHotVals, 0, PerceptionBuffer, cell_id*ObservationPerCell, ObservationPerCell);
                            /// </remarks>
                            float[] channelHotVals = new float[ObservationPerCell];
                            for (int j = 0; j < channelValues.Length; j++)
                            {
                                if (ChannelDepth[j] > 1)
                                {
                                    channelHotVals[(int)channelValues[j] + ChannelOffsets[j]] = 1f;
                                }
                                else
                                {
                                    channelHotVals[ChannelOffsets[j]] = channelValues[j];
                                }
                            }

                            Array.Copy(channelHotVals, 0, m_PerceptionBuffer, cellIndex * ObservationPerCell, ObservationPerCell);
                            break;
                    }
                    break;
                }
            }
        }

        /// <summary>Converts the index of the cell to the 3D point (y is zero)</summary>
        /// <returns>Vector3 of the position of the center of the cell</returns>
        /// <param name="cell">The index of the cell</param>
        /// <param name="shouldTransformPoint">Bool weather to transform the point to the current transform</param>
        protected Vector3 CellToPoint(int cell, bool shouldTransformPoint = true)
        {
            float x = (cell % GridNumSideZ - OffsetGridNumSide) * CellScaleX;
            float z = (cell / GridNumSideZ - OffsetGridNumSide) * CellScaleZ - DiffNumSideZX;

            if (shouldTransformPoint)
                return transform.TransformPoint(new Vector3(x, 0, z));
            return new Vector3(x, 0, z);
        }

        /// <summary>Finds the cell in which the given global point falls</summary>
        /// <returns>
        /// The index of the cell in which the global point falls or -1 if the point does not fall into a cell
        /// </returns>
        /// <param name="globalPoint">The 3D point in global space</param>
        public int PointToCell(Vector3 globalPoint)
        {
            Vector3 point = transform.InverseTransformPoint(globalPoint);

            if (point.x < -HalfOfGridX || point.x > HalfOfGridX || point.z < -HalfOfGridZ || point.z > HalfOfGridZ)
                return -1;

            float x = point.x + HalfOfGridX;
            float z = point.z + HalfOfGridZ;

            int _x = (int)Mathf.Floor(x * PointToCellScalingX);
            int _z = (int)Mathf.Floor(z * PointToCellScalingZ);
            return GridNumSideX * _z + _x;
        }

        /// <summary>Copies the data from one cell to another</summary>
        /// <param name="fromCellID">index of the cell to copy from</param>
        /// <param name="toCellID">index of the cell to copy into</param>
        protected void CopyCellData(int fromCellID, int toCellID)
        {
            Array.Copy(m_PerceptionBuffer,
                fromCellID * ObservationPerCell,
                m_PerceptionBuffer,
                toCellID * ObservationPerCell,
                ObservationPerCell);
            if (ShowGizmos)
                CellActivity[toCellID] = CellActivity[fromCellID];
        }

        /// <summary>Creates a copy of a float array</summary>
        /// <returns>float[] of the original data</returns>
        /// <param name="array">The array to copy from</parma>
        private static float[] CreateCopy(float[] array)
        {
            float[] b = new float[array.Length];
            System.Buffer.BlockCopy(array, 0, b, 0, array.Length * sizeof(float));
            return b;
        }

        /// <summary>Utility method to find the index of a tag</summary>
        /// <returns>Index of the tag in DetectableObjects, if it is in there</returns>
        /// <param name="tag">The tag to search for</param>
        public int IndexOfTag(string tag)
        {
            return Array.IndexOf(DetectableObjects, tag);
        }

        void OnDrawGizmos()
        {
            if (ShowGizmos)
            {
                if (Application.isEditor && !Application.isPlaying)
                    Start();

                Perceive();

                Vector3 scale = new Vector3(CellScaleX, 1, CellScaleZ);
                Vector3 offset = new Vector3(0, GizmoYOffset, 0);
                Matrix4x4 oldGizmoMatrix = Gizmos.matrix;
                Matrix4x4 cubeTransform = Gizmos.matrix;
                for (int i = 0; i < NumCells; i++)
                {
                    if (RotateToAgent)
                    {
                        cubeTransform = Matrix4x4.TRS(CellToPoint(i) + offset, transform.rotation, scale);
                    }
                    else
                    {
                        cubeTransform = Matrix4x4.TRS(CellToPoint(i, false) + transform.position + offset, Quaternion.identity, scale);
                    }
                    Gizmos.matrix = oldGizmoMatrix * cubeTransform;
                    Gizmos.color = CellActivity[i];
                    Gizmos.DrawCube(Vector3.zero, Vector3.one);
                }

                Gizmos.matrix = oldGizmoMatrix;

                if (Application.isEditor && !Application.isPlaying)
                    DestroyImmediate(m_perceptionTexture2D);
            }
        }

        /// <inheritdoc/>
        void ISensor.Update() { }

        /// <summary>Gets the observation shape</summary>
        /// <returns>int[] of the observation shape</returns>
        public override int[] GetObservationShape()
        {
            m_Shape = new[] { GridNumSideX, GridNumSideZ, ObservationPerCell };
            return m_Shape;
        }

        /// <inheritdoc/>
        public int Write(ObservationWriter writer)
        {
            // Timer stack is in accessable due to its protection level
            //    using (TimerStack.Instance.Scoped("GridSensor.WriteToTensor"))
            {
                Perceive();

                int index = 0;
                for (var h = GridNumSideZ - 1; h >= 0; h--) // height
                {
                    for (var w = 0; w < GridNumSideX; w++) // width
                    {
                        for (var d = 0; d < ObservationPerCell; d++) // depth
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
