using System.Collections.Generic;
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
        // dummy sensor only used for debug gizmo
        GridSensorBase m_DebugSensor;
        protected List<ISensor> m_Sensors;
        internal BoxOverlapChecker m_BoxOverlapChecker;

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
        internal Vector3Int m_GridSize = new Vector3Int(16, 1, 16);
        /// <summary>
        /// The number of grid on each side.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public Vector3Int GridSize
        {
            get { return m_GridSize; }
            set
            {
                if (value.y != 1)
                {
                    m_GridSize = new Vector3Int(value.x, 1, value.z);
                }
                else
                {
                    m_GridSize = value;
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
        internal string[] m_DetectableTags;
        /// <summary>
        /// List of tags that are detected.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public string[] DetectableTags
        {
            get { return m_DetectableTags; }
            set { m_DetectableTags = value; }
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

        [HideInInspector, SerializeField]
        internal bool m_UseOneHotTag = true;
        /// <summary>
        /// Whether to use one-hot detected tag as observation.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public bool UseOneHotTag
        {
            get { return m_UseOneHotTag; }
            set { m_UseOneHotTag = value; }
        }

        [HideInInspector, SerializeField]
        internal bool m_CountColliders = false;
        /// <summary>
        /// Whether to use the number of count for each detectable tag as observation.
        /// Note that changing this after the sensor is created has no effect.
        /// </summary>
        public bool CountColliders
        {
            get { return m_CountColliders; }
            set { m_CountColliders = value; }
        }

        /// <inheritdoc/>
        public override ISensor[] CreateSensors()
        {
            List<ISensor> m_Sensors = new List<ISensor>();
            m_BoxOverlapChecker = new BoxOverlapChecker(
                m_CellScale,
                m_GridSize,
                m_RotateWithAgent,
                m_ColliderMask,
                gameObject,
                m_DetectableTags,
                m_InitialColliderBufferSize,
                m_MaxColliderBufferSize
            );

            // debug data is positive int value and will trigger data validation exception if SensorCompressionType is not None.
            m_DebugSensor = new GridSensorBase("DebugGridSensor", m_CellScale, m_GridSize, m_DetectableTags, SensorCompressionType.None);
            m_BoxOverlapChecker.RegisterDebugSensor(m_DebugSensor);

            if (m_UseOneHotTag)
            {
                var sensor = new OneHotGridSensor(m_SensorName + "-OneHot", m_CellScale, m_GridSize, m_DetectableTags, m_CompressionType);
                if (ObservationStacks != 1)
                {
                    m_Sensors.Add(new StackingSensor(sensor, ObservationStacks));
                }
                else
                {
                    m_Sensors.Add(sensor);
                }
                m_BoxOverlapChecker.RegisterSensor(sensor);
            }
            if (m_CountColliders)
            {
                var sensor = new CountingGridSensor(m_SensorName + "-Counting", m_CellScale, m_GridSize, m_DetectableTags, m_CompressionType);
                if (ObservationStacks != 1)
                {
                    m_Sensors.Add(new StackingSensor(sensor, ObservationStacks));
                }
                else
                {
                    m_Sensors.Add(sensor);
                }
                m_BoxOverlapChecker.RegisterSensor(sensor);
            }
            var customSensors = GetGridSensors();
            if (customSensors != null)
            {
                foreach (var sensor in customSensors)
                {
                    if (ObservationStacks != 1)
                    {
                        m_Sensors.Add(new StackingSensor(sensor, ObservationStacks));
                    }
                    else
                    {
                        m_Sensors.Add(sensor);
                    }
                    m_BoxOverlapChecker.RegisterSensor(sensor);
                }
            }
            // Only one sensor needs to reference the boxOverlapChecker, so that it gets updated exactly once
            ((GridSensorBase)m_Sensors[0]).m_BoxOverlapChecker = m_BoxOverlapChecker;
            return m_Sensors.ToArray();
        }

        protected virtual GridSensorBase[] GetGridSensors()
        {
            return null;
        }

        /// <summary>
        /// Update fields that are safe to change on the Sensor at runtime.
        /// </summary>
        internal void UpdateSensor()
        {
            if (m_Sensors != null)
            {
                m_BoxOverlapChecker.RotateWithAgent = m_RotateWithAgent;
                m_BoxOverlapChecker.ColliderMask = m_ColliderMask;
                foreach (var sensor in m_Sensors)
                {
                    ((GridSensorBase)sensor).CompressionType = m_CompressionType;
                }
            }
        }

        void OnDrawGizmos()
        {
            if (m_ShowGizmos)
            {
                if (m_BoxOverlapChecker == null)
                {
                    return;
                }

                // hack for debug sensor: data is int value in [0, detectableTag.Length], so fill the buffer will -1 as default value.
                for (var i = 0; i < m_DebugSensor.PerceptionBuffer.Length; i++)
                {
                    m_DebugSensor.PerceptionBuffer[i] = -1f;
                }
                m_BoxOverlapChecker.UpdateGizmo();
                var cellColors = m_DebugSensor.PerceptionBuffer;
                var rotation = m_BoxOverlapChecker.GetGridRotation();

                var scale = new Vector3(m_CellScale.x, 1, m_CellScale.z);
                var gizmoYOffset = new Vector3(0, m_GizmoYOffset, 0);
                var oldGizmoMatrix = Gizmos.matrix;
                for (var i = 0; i < m_GridSize.x * m_GridSize.z; i++)
                {
                    var cellPosition = m_BoxOverlapChecker.GetCellGlobalPosition(i);
                    var cubeTransform = Matrix4x4.TRS(cellPosition + gizmoYOffset, rotation, scale);
                    Gizmos.matrix = oldGizmoMatrix * cubeTransform;
                    var colorIndex = cellColors[i];
                    var debugRayColor = Color.white;
                    if (colorIndex > -1 && m_DebugColors.Length > colorIndex)
                    {
                        debugRayColor = m_DebugColors[(int)colorIndex];
                    }
                    Gizmos.color = new Color(debugRayColor.r, debugRayColor.g, debugRayColor.b, .5f);
                    Gizmos.DrawCube(Vector3.zero, Vector3.one);
                }

                Gizmos.matrix = oldGizmoMatrix;
            }
        }
    }
}
