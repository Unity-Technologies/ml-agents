using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Areas
{
    /// <summary>
    /// The Training Ares Replicator allows for a training area object group to be replicated dynamically during runtime.
    /// </summary>
    [DefaultExecutionOrder(-100)]
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
        public float3 separation = new(10f, 10f, 10f);

        [FormerlySerializedAs("m_GridSize")]
        public int3 gridSize = new(1, 1, 1);

        /// <summary>
        /// Whether to replicate in the editor or in a build only. Default = true
        /// </summary>
        public bool buildOnly = true;

        int m_AreaCount;
        string m_TrainingAreaName;

        /// <summary>
        /// The size of the computed grid to pack the training areas into.
        /// </summary>
        public int3 GridSize => gridSize;

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

            if (numAreas == 0)
            {
                numAreas = gridSize.x * gridSize.y * gridSize.z;
            }
            else
            {
                var rootNumAreas = Mathf.Pow(numAreas - 0.5f, 1.0f / 3.0f);
                gridSize.x = Mathf.CeilToInt(rootNumAreas);
                gridSize.y = Mathf.CeilToInt(rootNumAreas);
                var zSize = Mathf.CeilToInt((float)numAreas / (gridSize.x * gridSize.y));
                gridSize.z = zSize == 0 ? 1 : zSize;
            }
        }

        /// <summary>
        /// Adds replicas of the training area to the scene.
        /// </summary>
        /// <exception cref="UnityAgentsException"></exception>
        void AddEnvironments()
        {
            baseArea.SetActive(false);

            if (numAreas > gridSize.x * gridSize.y * gridSize.z)
            {
                throw new UnityAgentsException("The number of training areas that you have specified exceeds the size of the grid.");
            }

            float zpos = -(gridSize.z - 1) * 0.5f * separation.z;

            GameObject zGameObj = null;
            for (int z = 0; z < gridSize.z; z++, zpos += separation.z)
            {
                zGameObj = new GameObject($"z{z}");
                zGameObj.transform.parent = this.transform;
                zGameObj.transform.localPosition = new Vector3(0, 0, zpos);
                float ypos = -(gridSize.y - 1) * 0.5f * separation.y;
                GameObject yGameObj = null;
                for (int y = 0; y < gridSize.y; y++, ypos += separation.y)
                {
                    yGameObj = new GameObject($"y{y}");
                    yGameObj.transform.parent = zGameObj.transform;
                    yGameObj.transform.localPosition = new Vector3(0, ypos, 0);
                    float xpos = -(gridSize.x - 1) * 0.5f * separation.x;
                    for (int x = 0; x < gridSize.x; x++, xpos += separation.x)
                    {
                        if (m_AreaCount < numAreas)
                        {
                            m_AreaCount++;
                            //var area = Instantiate(baseArea, new Vector3(0,0,0), Quaternion.identity);
                            var area = Instantiate(baseArea, yGameObj.transform);
                            area.transform.localPosition = new Vector3(xpos, 0, 0);
                            area.transform.parent = yGameObj.transform;
                            area.SetActive(true);
                            area.name = m_TrainingAreaName;
                        }
                    }
                }
            }
        }
    }
}
