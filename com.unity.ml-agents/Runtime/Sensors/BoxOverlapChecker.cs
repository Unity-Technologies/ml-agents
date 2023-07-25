using System;
using UnityEngine;

namespace Unity.MLAgents.Sensors
{
    /// <summary>
    /// The grid perception strategy that uses box overlap to detect objects.
    /// </summary>
    internal class BoxOverlapChecker : IGridPerception
    {
        Vector3 m_CellScale;
        Vector3Int m_GridSize;
        bool m_RotateWithAgent;
        LayerMask m_ColliderMask;
        GameObject m_CenterObject;
        GameObject m_AgentGameObject;
        string[] m_DetectableTags;
        int m_InitialColliderBufferSize;
        int m_MaxColliderBufferSize;

        int m_NumCells;
        Vector3 m_HalfCellScale;
        Vector3 m_CellCenterOffset;
        Vector3[] m_CellLocalPositions;

#if MLA_UNITY_PHYSICS_MODULE
        Collider[] m_ColliderBuffer;

        public event Action<GameObject, int> GridOverlapDetectedAll;
        public event Action<GameObject, int> GridOverlapDetectedClosest;
        public event Action<GameObject, int> GridOverlapDetectedDebug;
#endif

        public BoxOverlapChecker(
            Vector3 cellScale,
            Vector3Int gridSize,
            bool rotateWithAgent,
            LayerMask colliderMask,
            GameObject centerObject,
            GameObject agentGameObject,
            string[] detectableTags,
            int initialColliderBufferSize,
            int maxColliderBufferSize)
        {
            m_CellScale = cellScale;
            m_GridSize = gridSize;
            m_RotateWithAgent = rotateWithAgent;
            m_ColliderMask = colliderMask;
            m_CenterObject = centerObject;
            m_AgentGameObject = agentGameObject;
            m_DetectableTags = detectableTags;
            m_InitialColliderBufferSize = initialColliderBufferSize;
            m_MaxColliderBufferSize = maxColliderBufferSize;

            m_NumCells = gridSize.x * gridSize.z;
            m_HalfCellScale = new Vector3(cellScale.x / 2f, cellScale.y, cellScale.z / 2f);
            m_CellCenterOffset = new Vector3((gridSize.x - 1f) / 2, 0, (gridSize.z - 1f) / 2);
#if MLA_UNITY_PHYSICS_MODULE
            m_ColliderBuffer = new Collider[Math.Min(m_MaxColliderBufferSize, m_InitialColliderBufferSize)];
#endif

            InitCellLocalPositions();
        }

        public bool RotateWithAgent
        {
            get { return m_RotateWithAgent; }
            set { m_RotateWithAgent = value; }
        }

        public LayerMask ColliderMask
        {
            get { return m_ColliderMask; }
            set { m_ColliderMask = value; }
        }

        /// <summary>
        /// Initializes the local location of the cells
        /// </summary>
        void InitCellLocalPositions()
        {
            m_CellLocalPositions = new Vector3[m_NumCells];

            for (int i = 0; i < m_NumCells; i++)
            {
                m_CellLocalPositions[i] = GetCellLocalPosition(i);
            }
        }

        public Vector3 GetCellLocalPosition(int cellIndex)
        {
            float x = (cellIndex / m_GridSize.z - m_CellCenterOffset.x) * m_CellScale.x;
            float z = (cellIndex % m_GridSize.z - m_CellCenterOffset.z) * m_CellScale.z;
            return new Vector3(x, 0, z);
        }

        public Vector3 GetCellGlobalPosition(int cellIndex)
        {
            if (m_RotateWithAgent)
            {
                return m_CenterObject.transform.TransformPoint(m_CellLocalPositions[cellIndex]);
            }
            else
            {
                return m_CellLocalPositions[cellIndex] + m_CenterObject.transform.position;
            }
        }

        public Quaternion GetGridRotation()
        {
            return m_RotateWithAgent ? m_CenterObject.transform.rotation : Quaternion.identity;
        }

        public void Perceive()
        {
#if MLA_UNITY_PHYSICS_MODULE
            for (var cellIndex = 0; cellIndex < m_NumCells; cellIndex++)
            {
                var cellCenter = GetCellGlobalPosition(cellIndex);
                var numFound = BufferResizingOverlapBoxNonAlloc(cellCenter, m_HalfCellScale, GetGridRotation());

                if (GridOverlapDetectedAll != null)
                {
                    ParseCollidersAll(m_ColliderBuffer, numFound, cellIndex, cellCenter, GridOverlapDetectedAll);
                }
                if (GridOverlapDetectedClosest != null)
                {
                    ParseCollidersClosest(m_ColliderBuffer, numFound, cellIndex, cellCenter, GridOverlapDetectedClosest);
                }
            }
#endif
        }

        public void UpdateGizmo()
        {
#if MLA_UNITY_PHYSICS_MODULE
            for (var cellIndex = 0; cellIndex < m_NumCells; cellIndex++)
            {
                var cellCenter = GetCellGlobalPosition(cellIndex);
                var numFound = BufferResizingOverlapBoxNonAlloc(cellCenter, m_HalfCellScale, GetGridRotation());

                ParseCollidersClosest(m_ColliderBuffer, numFound, cellIndex, cellCenter, GridOverlapDetectedDebug);
            }
#endif
        }

#if MLA_UNITY_PHYSICS_MODULE
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
                numFound = Physics.OverlapBoxNonAlloc(cellCenter, halfCellScale, m_ColliderBuffer, rotation, m_ColliderMask);
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
        void ParseCollidersClosest(Collider[] foundColliders, int numFound, int cellIndex, Vector3 cellCenter, Action<GameObject, int> detectedAction)
        {
            GameObject closestColliderGo = null;
            var minDistanceSquared = float.MaxValue;

            for (var i = 0; i < numFound; i++)
            {
                var currentColliderGo = foundColliders[i].gameObject;

                // Continue if the current collider go is the root reference
                if (ReferenceEquals(currentColliderGo, m_AgentGameObject))
                {
                    continue;
                }

                var closestColliderPoint = foundColliders[i].ClosestPointOnBounds(cellCenter);
                var currentDistanceSquared = (closestColliderPoint - m_CenterObject.transform.position).sqrMagnitude;

                if (currentDistanceSquared >= minDistanceSquared)
                {
                    continue;
                }

                // Checks if our colliders contain a detectable object
                var index = -1;
                for (var ii = 0; ii < m_DetectableTags.Length; ii++)
                {
                    if (currentColliderGo.CompareTag(m_DetectableTags[ii]))
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
                detectedAction.Invoke(closestColliderGo, cellIndex);
            }
        }

        /// <summary>
        /// Parses all colliders in the array of colliders found within a cell.
        /// </summary>
        void ParseCollidersAll(Collider[] foundColliders, int numFound, int cellIndex, Vector3 cellCenter, Action<GameObject, int> detectedAction)
        {
            for (int i = 0; i < numFound; i++)
            {
                var currentColliderGo = foundColliders[i].gameObject;
                if (!ReferenceEquals(currentColliderGo, m_AgentGameObject))
                {
                    detectedAction.Invoke(currentColliderGo, cellIndex);
                }
            }
        }

#endif

        public void RegisterSensor(GridSensorBase sensor)
        {
#if MLA_UNITY_PHYSICS_MODULE
            if (sensor.GetProcessCollidersMethod() == ProcessCollidersMethod.ProcessAllColliders)
            {
                GridOverlapDetectedAll += sensor.ProcessDetectedObject;
            }
            else
            {
                GridOverlapDetectedClosest += sensor.ProcessDetectedObject;
            }
#endif
        }

        public void RegisterDebugSensor(GridSensorBase debugSensor)
        {
#if MLA_UNITY_PHYSICS_MODULE
            GridOverlapDetectedDebug += debugSensor.ProcessDetectedObject;
#endif
        }
    }
}
