using UnityEngine;

namespace Unity.MLAgentsExamples
{
    public class CameraFollow : MonoBehaviour
    {
        public Transform target;
        Vector3 m_Offset;

        private Vector3 m_CamVelocity;

        public float smoothingTime = .5f;
        // Use this for initialization
        void Start()
        {
            m_Offset = gameObject.transform.position - target.position;
        }

        // Update is called once per frame
        void Update()
        {
            var newPosition = new Vector3(target.position.x + m_Offset.x, transform.position.y,
                target.position.z + m_Offset.z);

            gameObject.transform.position = Vector3.SmoothDamp(transform.position, newPosition, ref m_CamVelocity, smoothingTime);
        }
    }
}
