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
        private float m_StartingYPos;
        
        void OnEnable()
        {
            m_StartingYPos = transform.position.y;
        }

        void Update()
        {
            transform.position = new Vector3(transformToFollow.position.x, m_StartingYPos + heightOffset, transformToFollow.position.z);
            Vector3 walkDir = targetToLookAt.position - transform.position;
            walkDir.y = 0; //flatten dir on the y
            transform.rotation = Quaternion.LookRotation(walkDir);
        }
    }
}
