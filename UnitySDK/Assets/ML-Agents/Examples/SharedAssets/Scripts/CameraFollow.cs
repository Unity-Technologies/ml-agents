using UnityEngine;

namespace MLAgents
{
    public class CameraFollow : MonoBehaviour
    {
        Transform m_Target;
        Vector3   m_Offset = new Vector3(5.0f, 7.0f, 5.0f);
        
        // Update is called once per frame
        void Update()
        {
            if (m_Target == null)
            {
                var targetGO = GameObject.Find("Body");
                m_Target = targetGO.transform;
                
            }
            
            gameObject.transform.position = m_Target.position + m_Offset;
            gameObject.transform.LookAt(m_Target);
        }
    }
}
