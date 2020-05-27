using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Unity.MLAgentsExamples
{
    public class DirectionIndicator : MonoBehaviour
    {
        public Transform transformToFollow; //ex: hips or body
        public Transform targetToLookAt; //target in the scene the indicator will point to
        public float heightOffset;
        private Vector3 m_StartingPos;
        
        void OnEnable()
        {
            m_StartingPos = transform.position;
        }

        void Update()
        {
            transform.position = new Vector3(transformToFollow.position.x, m_StartingPos.y + heightOffset, transformToFollow.position.z);
            Vector3 m_WalkDir = targetToLookAt.position - transform.position;
            m_WalkDir.y = 0; //flatten dir on the y
            transform.rotation = Quaternion.LookRotation(m_WalkDir);
        }
    }
}
