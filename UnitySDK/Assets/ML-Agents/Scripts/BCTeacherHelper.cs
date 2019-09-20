using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Behavioral Cloning Helper script. Attach to teacher agent to enable
    /// resetting the experience buffer, as well as toggling session recording.
    /// </summary>
    public class BcTeacherHelper : MonoBehaviour
    {
        bool m_RecordExperiences;
        bool m_ResetBuffer;
        Agent m_MyAgent;
        float m_BufferResetTime;

        public KeyCode recordKey = KeyCode.R;
        public KeyCode resetKey = KeyCode.C;

        // Use this for initialization
        void Start()
        {
            m_RecordExperiences = true;
            m_ResetBuffer = false;
            m_MyAgent = GetComponent<Agent>();
            m_BufferResetTime = Time.time;
        }

        // Update is called once per frame
        void Update()
        {
            if (Input.GetKeyDown(recordKey))
            {
                m_RecordExperiences = !m_RecordExperiences;
            }

            if (Input.GetKeyDown(resetKey))
            {
                m_ResetBuffer = true;
                m_BufferResetTime = Time.time;
            }
            else
            {
                m_ResetBuffer = false;
            }

            Monitor.Log("Recording experiences " + recordKey, m_RecordExperiences.ToString());
            var timeSinceBufferReset = Time.time - m_BufferResetTime;
            Monitor.Log("Seconds since buffer reset " + resetKey,
                Mathf.FloorToInt(timeSinceBufferReset).ToString());
        }

        void FixedUpdate()
        {
            // Convert both bools into single comma separated string. Python makes
            // assumption that this structure is preserved.
            m_MyAgent.SetTextObs(m_RecordExperiences + "," + m_ResetBuffer);
        }
    }
}
