using UnityEngine;

namespace MLAgentsExamples
{
    public class CameraFollow : MonoBehaviour
    {
        public Transform target;
        Vector3 m_Offset;

        // Use this for initialization
        void Start()
        {
            m_Offset = gameObject.transform.position - target.position;
        }

        // Update is called once per frame
        void Update()
        {
            // gameObject.transform.position = target.position + offset;
            var newPosition = new Vector3(target.position.x + m_Offset.x, transform.position.y,
                target.position.z + m_Offset.z);
            gameObject.transform.position = newPosition;
        }
    }
}
