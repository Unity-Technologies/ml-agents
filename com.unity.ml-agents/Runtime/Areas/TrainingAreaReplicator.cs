using System;
using Unity.Mathematics;
using UnityEngine;

namespace Unity.MLAgents.Areas
{
    public class TrainingAreaReplicator : MonoBehaviour
    {
        public GameObject baseArea;
        public int numAreas = 1;
        public float separation = 10f;

        int3 m_GridSize = new int3(1, 1, 1);
        int m_areaCount = 0;
        string m_TrainingAreaName;

        public int3 GridSize => m_GridSize;
        public string TrainingAreaName => m_TrainingAreaName;

        public void Awake()
        {
            ComputeGridSize();
            m_TrainingAreaName = baseArea.name;
        }

        public void OnEnable()
        {
            AddEnvironments();
        }

        void ComputeGridSize()
        {
            // check if running inference, if so, use the num areas set through the component,
            // otherwise, pull it from the academy
            if (Academy.Instance.Communicator != null)
                numAreas = Academy.Instance.NumAreas;

            var rootNumAreas = Mathf.Pow(numAreas, 1.0f / 3.0f);
            m_GridSize.x = Mathf.RoundToInt(rootNumAreas);
            m_GridSize.y = Mathf.RoundToInt(rootNumAreas);
            var zSize = numAreas - m_GridSize.x * m_GridSize.y;
            m_GridSize.z = zSize == 0 ? 1 : zSize;
        }

        void AddEnvironments()
        {
            if (numAreas > m_GridSize.x * m_GridSize.y * m_GridSize.z)
            {
                throw new UnityAgentsException("The number of training areas that you have specified exceeds the size of the grid.");
            }

            for (int z = 0; z < m_GridSize.z; z++)
            {
                for (int j = 0; j < m_GridSize.y; j++)
                {
                    for (int i = 0; i < m_GridSize.x; i++)
                    {
                        if (m_areaCount == 0)
                        {
                            m_areaCount = 1;
                        }
                        else if (m_areaCount < numAreas)
                        {
                            m_areaCount++;
                            var area = Instantiate(baseArea, new Vector3(i * separation, j * separation, z * separation), Quaternion.identity);
                            area.name = m_TrainingAreaName;
                        }
                    }
                }
            }
        }
    }
}

