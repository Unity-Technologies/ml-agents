using UnityEngine;

namespace Unity.MLAgentsExamples
{
    public class DirectionIndicator : MonoBehaviour
    {

        public bool updatedByAgent; //should this be updated by the agent? If not, it will use local settings
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
            if (updatedByAgent)
                return;
            transform.position = new Vector3(transformToFollow.position.x, m_StartingYPos + heightOffset,
                transformToFollow.position.z);
            Vector3 walkDir = targetToLookAt.position - transform.position;
            walkDir.y = 0; //flatten dir on the y
            transform.rotation = Quaternion.LookRotation(walkDir);
        }

        //Public method to allow an agent to directly update this component
        public void MatchOrientation(Transform t)
        {
            transform.position = new Vector3(t.position.x, m_StartingYPos + heightOffset, t.position.z);
            transform.rotation = t.rotation;
        }
    }
}
