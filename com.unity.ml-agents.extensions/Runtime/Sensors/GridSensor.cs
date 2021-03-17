using System;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.MLAgents.Sensors;
using UnityEngine.Profiling;
using Unity.Jobs;
using UnityEngine.Jobs;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// Grid-based sensor.
    /// </summary>
    [AddComponentMenu("ML Agents/Grid Sensor", (int)MenuGroup.Sensors)]
    public class GridSensor : SensorComponent, ISensor, IBuiltInSensor
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

        [Header("Collider Buffer Properties")]
        [Tooltip("The absolute max size of the Collider buffer used in the non-allocating Physics calls.  In other words" +
            " the Collider buffer will never grow beyond this number even if there are more Colliders in the Grid Cell.")]
        public int MaxColliderBufferSize = 500;
        [Tooltip(
            "The Estimated Max Number of Colliders to expect per cell.  This number is used to " +
            "pre-allocate an array of Colliders in order to take advantage of the OverlapBoxNonAlloc " +
            "Physics API.  If the number of colliders found is >= InitialColliderBufferSize the array " +
            "will be resized to double its current size.  The hard coded absolute size is 500.")]
        public int InitialColliderBufferSize = 4;
        // Collider[] m_ColliderBuffer;
        NativeArray<RaycastHit> m_Hits;
        NativeArray<int> m_HitIndexes;
        NativeArray<BoxcastCommand> m_BoxcastCommands;
        protected Dictionary<string, int> m_DetectableObjectToIndex = new Dictionary<string, int>();
        /// <summary>
        /// The offsets used to specify where within a cell's allotted data, certain observations will be inserted.
        /// </summary>
        [HideInInspector]
        public NativeArray<int> ChannelOffsets;

        public NativeArray<float> PerceptionBuffer => m_PerceptionBuffer;

        /// <summary>
        /// The main storage of perceptual information.
        /// </summary>
        protected NativeArray<float> m_PerceptionBuffer;

        /// <summary>
        ///  The default value of the perceptionBuffer when using the ChannelHot DepthType. Used to reset the array/
        /// </summary>
        protected NativeArray<float> m_ChannelHotDefaultPerceptionBuffer;

        /// <summary>
        /// Array of colors displaying the DebugColors for each cell in OnDrawGizmos. Only updated if ShowGizmos.
        /// </summary>
        protected NativeArray<Color> CellActivity;

        /// <summary>
        /// Array of positions where each position is the center of a cell.
        /// </summary>
        private NativeArray<Vector3> CellPoints;

        float[] m_ChannelBuffer;

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
        protected float InverseSphereRadius;

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

        public SensorCompressionType CompressionType = SensorCompressionType.PNG;

        /// <summary>
        /// List representing the multiple compressed images of all of the grids
        /// </summary>
        private List<byte[]> compressedImgs;

        /// <summary>
        /// List representing the sizes of the multiple images so they can be properly reconstructed on the python side
        /// </summary>
        private List<byte[]> byteSizesBytesList;

        private Color DebugDefaultColor = new Color(1f, 1f, 1f, 0.25f);
        internal JobHandle m_boxcastJobHandle;

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
            var sphereRadiusX = (CellScaleX * GridNumSideX) / Mathf.Sqrt(2);
            var sphereRadiusZ = (CellScaleZ * GridNumSideZ) / Mathf.Sqrt(2);
            InverseSphereRadius = 1.0f / Mathf.Max(sphereRadiusX, sphereRadiusZ);
            DisposeNativeArray(ChannelOffsets);
            ChannelOffsets = new NativeArray<int>(ChannelDepth.Length, Allocator.Persistent);
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

                    for (var i = 1; i < ChannelDepth.Length; i++)
                    {
                        ChannelOffsets[i] = ChannelOffsets[i - 1] + ChannelDepth[i - 1];
                    }

                    for (var i = 0; i < ChannelDepth.Length; i++)
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
            if (CellPoints.IsCreated)
            {
                CellPoints.Dispose();
            }

            CellPoints = new NativeArray<Vector3>(NumCells, Allocator.Persistent);

            for (var i = 0; i < NumCells; i++)
            {
                CellPoints[i] = CellToPoint(i, false);
            }
        }

        /// <summary>
        /// Initializes the m_ChannelHotDefaultPerceptionBuffer with default data in the case that the grid depth type is ChannelHot
        /// </summary>
        public virtual void InitChannelHotDefaultPerceptionBuffer()
        {
            if (m_ChannelHotDefaultPerceptionBuffer.IsCreated)
            {
                m_ChannelHotDefaultPerceptionBuffer.Dispose();
            }
            m_ChannelHotDefaultPerceptionBuffer = new NativeArray<float>(ObservationPerCell, Allocator.Persistent);
            for (var i = 0; i < ChannelDepth.Length; i++)
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
            if (m_PerceptionBuffer.IsCreated)
            {
                m_boxcastJobHandle.Complete();
            }
            DisposeNativeArray(m_PerceptionBuffer);
            m_PerceptionBuffer = new NativeArray<float>(NumberOfObservations, Allocator.Persistent);
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

            m_ChannelBuffer = new float[ChannelDepth.Length];
            DisposeNativeArray(CellActivity);
            CellActivity = new NativeArray<Color>(NumCells, Allocator.Persistent);
        }

        void OnEnable()
        {
            Start();
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
            DisposeNativeArray(m_Hits);
            m_Hits = new NativeArray<RaycastHit>(NumCells, Allocator.Persistent);

            DisposeNativeArray(m_HitIndexes);
            m_HitIndexes = new NativeArray<int>(NumCells, Allocator.Persistent);

            DisposeNativeArray(m_BoxcastCommands);
            m_BoxcastCommands = new NativeArray<BoxcastCommand>(NumCells, Allocator.Persistent);

            for (var i = 0; i < DetectableObjects.Length; i++)
            {
                m_DetectableObjectToIndex[DetectableObjects[i]] = i;
            }
            // Default root reference to current game object
            if (rootReference == null)
                rootReference = gameObject;
            m_Shape = new[] { GridNumSideX, GridNumSideZ, ObservationPerCell };

            compressedImgs = new List<byte[]>();
            byteSizesBytesList = new List<byte[]>();

            m_perceptionTexture2D = new Texture2D(GridNumSideX, GridNumSideZ, TextureFormat.RGB24, false);
        }

        /// <inheritdoc cref="ISensor.Reset"/>
        void ISensor.Reset() { }

        [BurstCompile(CompileSynchronously = true)]
        struct ClearBufferOneHotJob : IJobParallelFor
        {
            public NativeArray<float> PerceptionBuf;
            [ReadOnly]
            public NativeArray<float> ChannelHotBuf;
            public int ObservationPerCell;
            public void Execute(int index)
            {
                PerceptionBuf[index] = ChannelHotBuf[index % ObservationPerCell];
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct ClearBufferChannelJob : IJobParallelFor
        {
            public NativeArray<float> PerceptionBuf;
            public void Execute(int index)
            {
                PerceptionBuf[index] = 0;
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct ClearCellActivityJob : IJobParallelFor
        {
            public NativeArray<Color> CellActivity;
            public Color DebugDefault;
            public void Execute(int index)
            {
                CellActivity[index] = DebugDefault;
            }
        }

        /// <summary>
        /// Clears the perception buffer before loading in new data. If the gridDepthType is ChannelHot, then it initializes the
        /// Reset() also reinits the cell activity array (for debug)
        /// </summary>
        public void ClearPerceptionBuffer()
        {
            Profiler.BeginSample("ClearPerceptionBuffer");
            {

                if (gridDepthType == GridDepthType.ChannelHot)
                {
                    // Copy the default value to the array
                    for (var i = 0; i < m_PerceptionBuffer.Length; i++)
                    {
                        m_PerceptionBuffer[i] = m_ChannelHotDefaultPerceptionBuffer[i % ObservationPerCell];
                    }
                }
                else
                {
                    for (var i = 0; i < m_PerceptionBuffer.Length; i++)
                    {
                        m_PerceptionBuffer[i] = 0;
                    }
                }

                if (ShowGizmos)
                {
                    // Assign the default color to the cell activities
                    for (var i = 0; i < NumCells; i++)
                    {
                        CellActivity[i] = DebugDefaultColor;
                    }
                }
            }
            Profiler.EndSample();
        }

        void OnDisable()
        {
        }

        void OnDestroy()
        {
            DisposeNativeArray(m_Hits);
            DisposeNativeArray(m_HitIndexes);
            DisposeNativeArray(m_BoxcastCommands);
            DisposeNativeArray(CellActivity);
            DisposeNativeArray(CellPoints);
            DisposeNativeArray(m_ChannelHotDefaultPerceptionBuffer);
            DisposeNativeArray(ChannelOffsets);
        }

        static void DisposeNativeArray<TD>(NativeArray<TD> array)
        where TD : struct
        {
            if (array.IsCreated)
            {
                array.Dispose();
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
            return CompressionType;
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
            UpdateBufferFromJob();
            var allBytes = new List<byte>();
            Profiler.BeginSample("GridSensor.GetCompressedObservation");
            {

                for (var i = 0; i < NumImages - 1; i++)
                {
                    ChannelsToTexture(3 * i, 3);
                    allBytes.AddRange(m_perceptionTexture2D.EncodeToPNG());
                }

                ChannelsToTexture(3 * (NumImages - 1), NumChannelsOnLastImage);
                allBytes.AddRange(m_perceptionTexture2D.EncodeToPNG());
            }
            Profiler.EndSample();
            return allBytes.ToArray();
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
            for (var i = 0; i < NumCells; i++)
            {
                for (var j = 0; j < numChannelsToAdd; j++)
                {
                    m_PerceptionColors[i][j] = m_PerceptionBuffer[i * ObservationPerCell + channelIndex + j];
                }
            }
            m_perceptionTexture2D.SetPixels(m_PerceptionColors);
        }

        [BurstCompile(CompileSynchronously = true)]
        struct CreateBoxcastBatch : IJobParallelFor
        {
            public NativeArray<BoxcastCommand> Commands;
            public Vector3 halfScale;
            public int ObserveMask;
            [ReadOnly]
            public NativeArray<Vector3> CellPoints;
            public Vector3 Pos;

            public void Execute(int index)
            {
                var cellCenter = Pos + CellPoints[index];
                var rotation = Quaternion.identity;
                Commands[index] = new BoxcastCommand(new Vector3(cellCenter.x, cellCenter.y + 2.0f, cellCenter.z),
                    halfScale,
                    rotation,
                    Vector3.down,
                    10.0f,
                    ObserveMask);

            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct CreateBoxcastRotateBatch : IJobParallelFor
        {
            public NativeArray<BoxcastCommand> Commands;
            public Vector3 halfScale;
            public int ObserveMask;
            [ReadOnly]
            public NativeArray<Vector3> CellPoints;
            [ReadOnly]
            public Matrix4x4 Mat;
            public Quaternion Rotation;


            public void Execute(int index)
            {
                var cellCenter = Mat.MultiplyPoint(CellPoints[index]);
                var rotation = Rotation;
                Commands[index] = new BoxcastCommand(new Vector3(cellCenter.x, cellCenter.y + 7.0f, cellCenter.z),
                    halfScale,
                    rotation,
                    Vector3.down,
                    10.0f,
                    ObserveMask);

            }
        }

        // ReSharper disable Unity.PerformanceAnalysis
        /// <summary>
        /// Perceive - Clears the buffers, calls overlap box on the actual cell (the actual perception part)
        /// for all found colliders, LoadObjectData is called
        /// at the end, Perceive returns the float array of the perceptions
        /// </summary>
        /// <returns>A float[] containing all of the information collected from the gridsensor</returns>
        public NativeArray<float> Perceive()
        {
            if (!m_PerceptionBuffer.IsCreated || !m_Hits.IsCreated || !m_BoxcastCommands.IsCreated)
            {
                return new NativeArray<float>();
            }

            Profiler.BeginSample("GridSensor.Perceive");
            {
                m_boxcastJobHandle.Complete();
                JobHandle clearHandle;
                if (gridDepthType == GridDepthType.ChannelHot)
                {
                    var clearJob = new ClearBufferOneHotJob
                    {
                        PerceptionBuf = m_PerceptionBuffer,
                        ChannelHotBuf = m_ChannelHotDefaultPerceptionBuffer,
                        ObservationPerCell = ObservationPerCell
                    };
                    clearHandle = clearJob.Schedule(NumberOfObservations, NumberOfObservations / 12);

                }
                else
                {
                    var clearJob = new ClearBufferChannelJob
                    {
                        PerceptionBuf = m_PerceptionBuffer
                    };
                    clearHandle = clearJob.Schedule(NumberOfObservations, NumberOfObservations / 12);
                }

                if (ShowGizmos)
                {
                    var gizmoJob = new ClearCellActivityJob
                    {
                        CellActivity = CellActivity,
                        DebugDefault = DebugDefaultColor
                    };
                    var gizmoHandle = gizmoJob.Schedule(NumCells, NumCells / 12);
                    clearHandle = JobHandle.CombineDependencies(clearHandle, gizmoHandle);
                }

                var t = transform;
                var halfCellScale = new Vector3(CellScaleX / 2f, CellScaleY / 2f, CellScaleZ / 2f);
                JobHandle createBoxHandle;
                if (RotateToAgent)
                {
                    var createBoxcasts = new CreateBoxcastRotateBatch
                    {
                        Commands = m_BoxcastCommands,
                        halfScale = halfCellScale,
                        Mat = t.localToWorldMatrix,
                        CellPoints = CellPoints,
                        ObserveMask = ObserveMask,
                        Rotation = t.rotation
                    };
                    createBoxHandle = createBoxcasts.Schedule(NumCells, NumCells / 12, clearHandle);
                }
                else
                {

                    var createBoxcasts = new CreateBoxcastBatch
                    {
                        Commands = m_BoxcastCommands,
                        halfScale = halfCellScale,
                        Pos = t.position,
                        CellPoints = CellPoints,
                        ObserveMask = ObserveMask
                    };
                    createBoxHandle = createBoxcasts.Schedule(NumCells, NumCells / 12, clearHandle);
                }

                m_boxcastJobHandle = BoxcastCommand.ScheduleBatch(m_BoxcastCommands, m_Hits, NumCells / 12, createBoxHandle);

            }
            Profiler.EndSample();
            return m_PerceptionBuffer;
        }


        /// <summary>
        /// Parses the array of colliders found within a cell. Finds the closest gameobject to the agent root reference within the cell
        /// </summary>
        /// <param name="foundColliders">Array of the colliders found within the cell</param>
        /// <param name="numFound">Number of colliders found.</param>
        /// <param name="cellIndex">The index of the cell</param>
        /// <param name="cellCenter">The center position of the cell</param>
        protected virtual void ParseColliders(Collider[] foundColliders, int numFound, int cellIndex, Vector3 cellCenter)
        {
            // Profiler.BeginSample("GridSensor.ParseColliders");
            // GameObject closestColliderGo = null;
            // var minDistanceSquared = float.MaxValue;
            //
            // var detectableIndex = -1;
            // for (var i = 0; i < numFound; i++)
            // {
            //     if (ReferenceEquals(foundColliders[i], null))
            //     {
            //         continue;
            //     }
            //     var currentColliderGo = foundColliders[i].gameObject;
            //
            //     // Continue if the current collider go is the root reference
            //     if (ReferenceEquals(currentColliderGo, rootReference))
            //         continue;
            //
            //     var closestColliderPoint = foundColliders[i].ClosestPointOnBounds(cellCenter);
            //     var currentDistanceSquared = (closestColliderPoint - rootReference.transform.position).sqrMagnitude;
            //     Profiler.EndSample();
            //
            //     // Checks if our colliders contain a detectable object
            //     if (m_DetectableObjectToIndex.TryGetValue(currentColliderGo.tag, out detectableIndex) && currentDistanceSquared < minDistanceSquared)
            //     {
            //     }
            //     for (var ii = 0; ii < DetectableObjects.Length; ii++)
            //     {
            //         if (currentColliderGo.CompareTag(DetectableObjects[ii]))
            //         {
            //             minDistanceSquared = currentDistanceSquared;
            //             closestColliderGo = currentColliderGo;
            //             break;
            //         }
            //     }
            // }
            //
            // if (!ReferenceEquals(closestColliderGo, null))
            //     LoadObjectData(closestColliderGo, cellIndex, detectableIndex, (float)Math.Sqrt(minDistanceSquared) * InverseSphereRadius);
            // Profiler.EndSample();
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
            for (var j = 0; j < channelValues.Length; j++)
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
        /// <param name="detectableIndex">Index into the DetectableObjects array.</param>
        /// <param name="normalizedDistance">A float between 0 and 1 describing the ratio of
        ///     the distance currentColliderGo is compared to the edge of the gridsensor</param>
        protected virtual void LoadObjectData(GameObject currentColliderGo, int cellIndex, int detectableIndex, float normalizedDistance)
        {
            Profiler.BeginSample("GridSensor.LoadObjectData");
            var offset = cellIndex * ObservationPerCell;
            for (var ii = 0; ii < ObservationPerCell; ii++)
            {
                m_PerceptionBuffer[offset + ii] = 0f;
            }

            // TODO: Create the array already then set the values using "out" in GetObjectData
            // Using i+1 as the type index as "0" represents "empty"
            var channelValues = GetObjectData(currentColliderGo, (float)detectableIndex + 1, normalizedDistance);

            ValidateValues(channelValues, currentColliderGo);
            if (ShowGizmos)
            {
                var debugRayColor = Color.white;
                if (DebugColors.Length > 0)
                {
                    debugRayColor = DebugColors[detectableIndex];
                }

                CellActivity[cellIndex] = new Color(debugRayColor.r, debugRayColor.g, debugRayColor.b, .5f);
            }

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
                        for (var j = 0; j < channelValues.Length; j++)
                        {
                            channelValues[j] /= ChannelDepth[j];
                        }

                        for (var ii = 0; ii < ObservationPerCell; ii++)
                        {
                            m_PerceptionBuffer[offset + ii] = channelValues[ii];
                        }
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
                        for (var j = 0; j < channelValues.Length; j++)
                        {
                            if (ChannelDepth[j] > 1)
                            {
                                m_PerceptionBuffer[offset + (int)channelValues[j] + ChannelOffsets[j]] = 1f;
                            }
                            else
                            {
                                m_PerceptionBuffer[offset + ChannelOffsets[j]] = channelValues[j];
                            }
                        }

                        break;
                    }
            }

            Profiler.EndSample();
        }

        /// <summary>Converts the index of the cell to the 3D point (y is zero)</summary>
        /// <returns>Vector3 of the position of the center of the cell</returns>
        /// <param name="cell">The index of the cell</param>
        /// <param name="shouldTransformPoint">Bool weather to transform the point to the current transform</param>
        protected Vector3 CellToPoint(int cell, bool shouldTransformPoint = true)
        {
            var x = (cell % GridNumSideZ - OffsetGridNumSide) * CellScaleX;
            var z = (cell / GridNumSideZ - OffsetGridNumSide) * CellScaleZ - DiffNumSideZX;

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
            var point = transform.InverseTransformPoint(globalPoint);

            if (point.x < -HalfOfGridX || point.x > HalfOfGridX || point.z < -HalfOfGridZ || point.z > HalfOfGridZ)
                return -1;

            var x = point.x + HalfOfGridX;
            var z = point.z + HalfOfGridZ;

            var _x = (int)Mathf.Floor(x * PointToCellScalingX);
            var _z = (int)Mathf.Floor(z * PointToCellScalingZ);
            return GridNumSideX * _z + _x;
        }

        // /// <summary>Copies the data from one cell to another</summary>
        // /// <param name="fromCellID">index of the cell to copy from</param>
        // /// <param name="toCellID">index of the cell to copy into</param>
        // protected void CopyCellData(int fromCellID, int toCellID)
        // {
        //     Array.Copy(m_PerceptionBuffer,
        //         fromCellID * ObservationPerCell,
        //         m_PerceptionBuffer,
        //         toCellID * ObservationPerCell,
        //         ObservationPerCell);
        //     if (ShowGizmos)
        //         CellActivity[toCellID] = CellActivity[fromCellID];
        // }

        void OnDrawGizmos()
        {
            if (ShowGizmos)
            {
                if (Application.isEditor && !Application.isPlaying)
                    Start();

                Perceive();
                UpdateBufferFromJob();

                var scale = new Vector3(CellScaleX, CellScaleY, CellScaleZ);
                var offset = new Vector3(0, 7.0f, 0);
                var oldGizmoMatrix = Gizmos.matrix;
                for (var i = 0; i < NumCells; i++)
                {
                    Matrix4x4 cubeTransform;
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
                    Gizmos.DrawWireCube(Vector3.zero, Vector3.one);
                }

                Gizmos.matrix = oldGizmoMatrix;

                if (Application.isEditor && !Application.isPlaying)
                    DestroyImmediate(m_perceptionTexture2D);
            }
        }

        /// <inheritdoc/>
        void ISensor.Update()
        {
            Profiler.BeginSample("GridSensor.Update");
            {
                Perceive();
            }
            Profiler.EndSample();
        }

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
            UpdateBufferFromJob();
            var index = 0;
            Profiler.BeginSample("GridSensor.WriteToTensor");
            {
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
                Profiler.EndSample();
            }
            return index;
        }

        internal void UpdateBufferFromJob()
        {
            Profiler.BeginSample("UpdateBufferFromJob");
            {
                m_boxcastJobHandle.Complete();
                for (var cellIndex = 0; cellIndex < NumCells; cellIndex++)
                {
                    var c = m_Hits[cellIndex].collider;
                    if (ReferenceEquals(c, null))
                    {
                        continue;
                    }

                    for (var i = 0; i < DetectableObjects.Length; i++)
                    {
                        if (c.CompareTag(DetectableObjects[i]))
                        {
                            LoadObjectData(c.gameObject, cellIndex, i, 0);
                            break;
                        }
                    }
                }
            }
            Profiler.EndSample();
        }
    }
}
