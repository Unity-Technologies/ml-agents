using UnityEngine;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Extensions.Sensors
{
    /// <summary>
    /// A SensorComponent that creates a <see cref="GridSensor"/>.
    /// </summary>
    [AddComponentMenu("ML Agents/Grid Sensor", (int)MenuGroup.Sensors)]
    public class GridSensorComponent : SensorComponent
    {
        GridSensor m_Sensor;

        [HideInInspector, SerializeField]
        string m_SensorName = "GridSensor";
        // <summary>
        /// Name of the generated <see cref="GridSensor"/> object.
        /// Note that changing this at runtime does not affect how the Agent sorts the sensors.
        /// </summary>
        public string SensorName
        {
            get { return m_SensorName; }
            set { m_SensorName = value; }
        }

        [HideInInspector, SerializeField]
        [Range(0.05f, 1000f)]
        float m_CellScaleX = 1f;

        /// <summary>
        /// The width of each grid cell.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public float CellScaleX
        {
            get { return m_CellScaleX; }
            set { m_CellScaleX = value; }
        }

        [HideInInspector, SerializeField]
        [Range(0.05f, 1000f)]
        float m_CellScaleZ = 1f;
        /// <summary>
        /// The depth of each grid cell.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public float CellScaleZ
        {
            get { return m_CellScaleZ; }
            set { m_CellScaleZ = value; }
        }

        [HideInInspector, SerializeField]
        [Range(0.01f, 1000f)]
        float m_CellScaleY = 0.01f;
        /// <summary>
        /// The height of each grid cell. Changes how much of the vertical axis is observed by a cell.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public float CellScaleY
        {
            get { return m_CellScaleY; }
            set { m_CellScaleY = value; }
        }

        [HideInInspector, SerializeField]
        [Range(2, 2000)]
        int m_GridNumSideX = 16;
        /// <summary>
        /// The width of the grid .
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public int GridNumSideX
        {
            get { return m_GridNumSideX; }
            set { m_GridNumSideX = value; }
        }

        [HideInInspector, SerializeField]
        [Range(2, 2000)]
        int m_GridNumSideZ = 16;
        /// <summary>
        /// The depth of the grid .
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public int GridNumSideZ
        {
            get { return m_GridNumSideZ; }
            set { m_GridNumSideZ = value; }
        }

        [HideInInspector, SerializeField]
        bool m_RotateWithAgent = true;
        /// <summary>
        /// Rotate the grid based on the direction the agent is facing.
        /// </summary>
        public bool RotateWithAgent
        {
            get { return m_RotateWithAgent; }
            set { m_RotateWithAgent = value; }
        }

        [HideInInspector, SerializeField]
        int[] m_ChannelDepth = new int[] { 1 };
        /// <summary>
        /// Array holding the depth of each channel.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public int[] ChannelDepth
        {
            get { return m_ChannelDepth; }
            set { m_ChannelDepth = value; }
        }

        [HideInInspector, SerializeField]
        string[] m_DetectableObjects;
        /// <summary>
        /// List of tags that are detected.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public string[] DetectableObjects
        {
            get { return m_DetectableObjects; }
            set { m_DetectableObjects = value; }
        }

        [HideInInspector, SerializeField]
        LayerMask m_ObserveMask;
        /// <summary>
        /// The layer mask.
        /// </summary>
        public LayerMask ObserveMask
        {
            get { return m_ObserveMask; }
            set { m_ObserveMask = value; }
        }

        [HideInInspector, SerializeField]
        GridDepthType m_DepthType = GridDepthType.Channel;
        /// <summary>
        /// The data layout that the grid should output.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public GridDepthType DepthType
        {
            get { return m_DepthType; }
            set { m_DepthType = value; }
        }

        [HideInInspector, SerializeField]
        GameObject m_RootReference;
        /// <summary>
        /// The reference of the root of the agent. This is used to disambiguate objects with the same tag as the agent. Defaults to current GameObject.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public GameObject RootReference
        {
            get { return m_RootReference == null ? gameObject : m_RootReference; }
            set { m_RootReference = value; }
        }

        [HideInInspector, SerializeField]
        int m_MaxColliderBufferSize = 500;
        /// <summary>
        /// The absolute max size of the Collider buffer used in the non-allocating Physics calls.  In other words
        /// the Collider buffer will never grow beyond this number even if there are more Colliders in the Grid Cell.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public int MaxColliderBufferSize
        {
            get { return m_MaxColliderBufferSize; }
            set { m_MaxColliderBufferSize = value; }
        }

        [HideInInspector, SerializeField]
        int m_InitialColliderBufferSize = 4;
        /// <summary>
        /// The Estimated Max Number of Colliders to expect per cell.  This number is used to
        /// pre-allocate an array of Colliders in order to take advantage of the OverlapBoxNonAlloc
        /// Physics API.  If the number of colliders found is >= InitialColliderBufferSize the array
        /// will be resized to double its current size.  The hard coded absolute size is 500.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public int InitialColliderBufferSize
        {
            get { return m_InitialColliderBufferSize; }
            set { m_InitialColliderBufferSize = value; }
        }

        [HideInInspector, SerializeField]
        Color[] m_DebugColors;
        /// <summary>
        /// Array of Colors used for the grid gizmos.
        /// </summary>
        public Color[] DebugColors
        {
            get { return m_DebugColors; }
            set { m_DebugColors = value; }
        }

        [HideInInspector, SerializeField]
        float m_GizmoYOffset = 0f;
        /// <summary>
        /// The height of the gizmos grid.
        /// </summary>
        public float GizmoYOffset
        {
            get { return m_GizmoYOffset; }
            set { m_GizmoYOffset = value; }
        }

        [HideInInspector, SerializeField]
        bool m_ShowGizmos = false;
        /// <summary>
        /// Whether to show gizmos or not.
        /// </summary>
        public bool ShowGizmos
        {
            get { return m_ShowGizmos; }
            set { m_ShowGizmos = value; }
        }

        [HideInInspector, SerializeField]
        SensorCompressionType m_CompressionType = SensorCompressionType.PNG;
        /// <summary>
        /// The compression type to use for the sensor.
        /// </summary>
        public SensorCompressionType CompressionType
        {
            get { return m_CompressionType; }
            set { m_CompressionType = value; UpdateSensor(); }
        }

        [HideInInspector, SerializeField]
        [Range(1, 50)]
        [Tooltip("Number of frames of observations that will be stacked before being fed to the neural network.")]
        int m_ObservationStacks = 1;
        /// <summary>
        /// Whether to stack previous observations. Using 1 means no previous observations.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public int ObservationStacks
        {
            get { return m_ObservationStacks; }
            set { m_ObservationStacks = value; }
        }

        /// <inheritdoc/>
        public override ISensor[] CreateSensors()
        {
            m_Sensor = new GridSensor(
                m_SensorName,
                m_CellScaleX,
                m_CellScaleY,
                m_CellScaleZ,
                m_GridNumSideX,
                m_GridNumSideZ,
                m_RotateWithAgent,
                m_ChannelDepth,
                m_DetectableObjects,
                m_ObserveMask,
                m_DepthType,
                RootReference,
                m_CompressionType,
                m_MaxColliderBufferSize,
                m_InitialColliderBufferSize,
                m_ShowGizmos,
                m_DebugColors
            );

            if (ObservationStacks != 1)
            {
                return new ISensor[] { new StackingSensor(m_Sensor, ObservationStacks) };
            }
            return new ISensor[] { m_Sensor };
        }

        /// <summary>
        /// Update fields that are safe to change on the Sensor at runtime.
        /// </summary>
        internal void UpdateSensor()
        {
            if (m_Sensor != null)
            {
                m_Sensor.CompressionType = m_CompressionType;
                m_Sensor.ShowGizmos = m_ShowGizmos;
                m_Sensor.DebugColors = (Color[])m_DebugColors.Clone();
            }
        }

        void OnDrawGizmos()
        {
            if (m_ShowGizmos)
            {
                if (m_Sensor == null)
                {
                    return;
                }
                m_Sensor.Perceive();

                var scale = new Vector3(m_CellScaleX, 1, m_CellScaleZ);
                var gizmoYOffset = new Vector3(0, m_GizmoYOffset, 0);
                var oldGizmoMatrix = Gizmos.matrix;
                for (var i = 0; i < m_GridNumSideX * m_GridNumSideZ; i++)
                {
                    Matrix4x4 cubeTransform;
                    var cellPoints = m_Sensor.GetCellGlobalPosition(i);
                    cubeTransform = Matrix4x4.TRS(cellPoints + gizmoYOffset, m_Sensor.GetGridRotation(), scale);
                    Gizmos.matrix = oldGizmoMatrix * cubeTransform;
                    var colorIndex = m_Sensor.CellActivity[i];
                    var debugRayColor = Color.white;
                    if (colorIndex >= 0 && m_DebugColors.Length > colorIndex)
                    {
                        debugRayColor = m_DebugColors[colorIndex];
                    }
                    Gizmos.color = new Color(debugRayColor.r, debugRayColor.g, debugRayColor.b, .5f);
                    Gizmos.DrawCube(Vector3.zero, Vector3.one);
                }

                Gizmos.matrix = oldGizmoMatrix;
            }
        }
    }
}
