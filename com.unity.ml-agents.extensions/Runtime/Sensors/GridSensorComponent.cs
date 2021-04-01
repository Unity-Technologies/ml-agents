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
        protected GridSensor m_Sensor;

        [HideInInspector, SerializeField]
        internal string m_SensorName = "GridSensor";
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
        internal Vector3 m_CellScale = new Vector3(1f, 0.01f, 1f);

        /// <summary>
        /// The scale of each grid cell.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public Vector3 CellScale
        {
            get { return m_CellScale; }
            set { m_CellScale = value; }
        }

        [HideInInspector, SerializeField]
        internal Vector3Int m_GridNum = new Vector3Int(16, 1, 16);
        /// <summary>
        /// The number of grid on each side.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public Vector3Int GridNum
        {
            get { return m_GridNum; }
            set
            {
                if (value.y != 1)
                {
                    m_GridNum = new Vector3Int(value.x, 1, value.z);
                }
                else
                {
                    m_GridNum = value;
                }
            }
        }

        [HideInInspector, SerializeField]
        internal bool m_RotateWithAgent = true;
        /// <summary>
        /// Rotate the grid based on the direction the agent is facing.
        /// </summary>
        public bool RotateWithAgent
        {
            get { return m_RotateWithAgent; }
            set { m_RotateWithAgent = value; }
        }

        [HideInInspector, SerializeField]
        internal int[] m_ChannelDepths = new int[] { 1 };
        /// <summary>
        /// Array holding the depth of each channel.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public int[] ChannelDepths
        {
            get { return m_ChannelDepths; }
            set { m_ChannelDepths = value; }
        }

        [HideInInspector, SerializeField]
        internal string[] m_DetectableObjects;
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
        internal LayerMask m_ColliderMask;
        /// <summary>
        /// The layer mask.
        /// </summary>
        public LayerMask ColliderMask
        {
            get { return m_ColliderMask; }
            set { m_ColliderMask = value; }
        }

        [HideInInspector, SerializeField]
        internal GridDepthType m_DepthType = GridDepthType.Channel;
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
        internal GameObject m_RootReference;
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
        internal int m_MaxColliderBufferSize = 500;
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
        internal int m_InitialColliderBufferSize = 4;
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
        internal Color[] m_DebugColors;
        /// <summary>
        /// Array of Colors used for the grid gizmos.
        /// </summary>
        public Color[] DebugColors
        {
            get { return m_DebugColors; }
            set { m_DebugColors = value; }
        }

        [HideInInspector, SerializeField]
        internal float m_GizmoYOffset = 0f;
        /// <summary>
        /// The height of the gizmos grid.
        /// </summary>
        public float GizmoYOffset
        {
            get { return m_GizmoYOffset; }
            set { m_GizmoYOffset = value; }
        }

        [HideInInspector, SerializeField]
        internal bool m_ShowGizmos = false;
        /// <summary>
        /// Whether to show gizmos or not.
        /// </summary>
        public bool ShowGizmos
        {
            get { return m_ShowGizmos; }
            set { m_ShowGizmos = value; }
        }

        [HideInInspector, SerializeField]
        internal SensorCompressionType m_CompressionType = SensorCompressionType.PNG;
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
        internal int m_ObservationStacks = 1;
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
                m_CellScale,
                m_GridNum,
                m_RotateWithAgent,
                m_ChannelDepths,
                m_DetectableObjects,
                m_ColliderMask,
                m_DepthType,
                RootReference,
                m_CompressionType,
                m_MaxColliderBufferSize,
                m_InitialColliderBufferSize
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
                m_Sensor.RotateWithAgent = m_RotateWithAgent;
                m_Sensor.ColliderMask = m_ColliderMask;
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
                var cellColors = m_Sensor.PerceiveGizmoColor();
                var cellPositions = m_Sensor.GetGizmoPositions();
                var rotation = m_Sensor.GetGridRotation();

                var scale = new Vector3(m_CellScale.x, 1, m_CellScale.z);
                var gizmoYOffset = new Vector3(0, m_GizmoYOffset, 0);
                var oldGizmoMatrix = Gizmos.matrix;
                for (var i = 0; i < cellPositions.Length; i++)
                {
                    var cubeTransform = Matrix4x4.TRS(cellPositions[i] + gizmoYOffset, rotation, scale);
                    Gizmos.matrix = oldGizmoMatrix * cubeTransform;
                    var colorIndex = cellColors[i];
                    var debugRayColor = Color.white;
                    if (colorIndex > -1 && m_DebugColors.Length > colorIndex)
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
