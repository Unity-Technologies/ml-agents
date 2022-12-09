using System;
using Unity.Mathematics;
using UnityEngine;

namespace Unity.MLAgents.Areas
{
    /// <summary>
    /// The Training Ares Replicator allows for a training area object group to be replicated dynamically during runtime.
    /// </summary>
    [DefaultExecutionOrder(-5)]
    public class TrainingAreaReplicator : MonoBehaviour
    {
        /// <summary>
        /// The base training area to be replicated.
        /// </summary>
        public GameObject baseArea;

        /// <summary>
        /// The number of training areas to replicate.
        /// </summary>
        public int numAreas = 1;

        /// <summary>
        /// The separation between each training area.
        /// </summary>
        public float separation = 10f;

        /// <summary>
        /// Whether to replicate in the editor or in a build only. Default = true
        /// </summary>
        public bool buildOnly = true;

        int3 m_GridSize = new(1, 1, 1);
        int m_AreaCount;
        string m_TrainingAreaName;

        /// <summary>
        /// The size of the computed grid to pack the training areas into.
        /// </summary>
        public int3 GridSize => m_GridSize;

        /// <summary>
        /// The name of the training area.
        /// </summary>
        public string TrainingAreaName => m_TrainingAreaName;

        /// <summary>
        /// Called before the simulation begins to computed the grid size for distributing
        /// the replicated training areas and set the area name.
        /// </summary>
        public void Awake()
        {
            // Computes the Grid Size on Awake
            ComputeGridSize();
            // Sets the TrainingArea name to the name of the base area.
            m_TrainingAreaName = baseArea.name;
        }

        /// <summary>
        /// Called after Awake and before the simulation begins and adds the training areas before
        /// the Academy begins.
        /// </summary>
        public void OnEnable()
        {
            // Adds the training as replicas during OnEnable to ensure they are added before the Academy begins its work.
            if (buildOnly)
            {
#if UNITY_STANDALONE && !UNITY_EDITOR
                AddEnvironments();
#endif
                return;
            }
            AddEnvironments();
        }

        /// <summary>
        /// Computes the Grid Size for replicating the training area.
        /// </summary>
        void ComputeGridSize()
        {
            // check if running inference, if so, use the num areas set through the component,
            // otherwise, pull it from the academy
            if (Academy.Instance.Communicator != null)
                numAreas = Academy.Instance.NumAreas;

            var rootNumAreas = Mathf.Pow(numAreas, 1.0f / 3.0f);
            m_GridSize.x = Mathf.CeilToInt(rootNumAreas);
            m_GridSize.y = Mathf.CeilToInt(rootNumAreas);
            var zSize = Mathf.CeilToInt((float)numAreas / (m_GridSize.x * m_GridSize.y));
            m_GridSize.z = zSize == 0 ? 1 : zSize;
        }

        /// <summary>
        /// Adds replicas of the training area to the scene.
        /// </summary>
        /// <exception cref="UnityAgentsException"></exception>
        void AddEnvironments()
        {
            if (numAreas > m_GridSize.x * m_GridSize.y * m_GridSize.z)
            {
                throw new UnityAgentsException("The number of training areas that you have specified exceeds the size of the grid.");
            }

            for (int z = 0; z < m_GridSize.z; z++)
            {
                for (int y = 0; y < m_GridSize.y; y++)
                {
                    for (int x = 0; x < m_GridSize.x; x++)
                    {
                        if (m_AreaCount == 0)
                        {
                            // Skip this first area since it already exists.
                            m_AreaCount = 1;
                        }
                        else if (m_AreaCount < numAreas)
                        {
                            m_AreaCount++;
                            var area = Instantiate(baseArea, new Vector3(x * separation, y * separation, z * separation), Quaternion.identity);
                            area.name = m_TrainingAreaName;
                        }
                    }
                }
            }
        }
    }
}
