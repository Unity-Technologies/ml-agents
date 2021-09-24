using System;
using Unity.Mathematics;
using UnityEngine;

namespace Unity.MLAgents.Areas
{
    public class TrainingAreaReplicator : MonoBehaviour
    {
        public GameObject baseArea;
        public int numAreas = 0;
        public float separation = 0f;
        public int3 gridSize = new int3(1, 1, 1);

        int m_areaCount = 0;

        public void Awake()
        {
            AddEnvironments();
            // Academy.Instance.OnEnvironmentReset += AddEnvironments;
        }

        void AddEnvironments()
        {
            // check if running inference, if so, use the num areas set through the component,
            // otherwise, pull it from the academy
            if (Academy.Instance.Communicator != null)
                numAreas = Academy.Instance.NumAreas;

            if (numAreas > gridSize.x * gridSize.y * gridSize.z)
            {
                throw new UnityAgentsException("The number of training areas that you have specified exceeds the size of the grid.");
            }

            for (int z = 0; z < gridSize.z; z++)
            {
                for (int j = 0; j < gridSize.y; j++)
                {
                    for (int i = 0; i < gridSize.x; i++)
                    {
                        if (m_areaCount == 0)
                        {
                            m_areaCount = 1;
                        }
                        else if (m_areaCount < numAreas)
                        {
                            m_areaCount++;
                            var area = Instantiate(baseArea, new Vector3(i * separation, j * separation, z * separation), Quaternion.identity);
                            // StartAllObjects(area);
                        }

                    }
                }
            }

            /*for (int i = 0; i < numAreas - 1; i++)
            {
                Instantiate(baseArea, new Vector3(0, 0, (i + 1) * separation), Quaternion.identity);
            }*/
        }

        void StartAllObjects(GameObject area)
        {
            throw new NotImplementedException();
        }
    }
}

