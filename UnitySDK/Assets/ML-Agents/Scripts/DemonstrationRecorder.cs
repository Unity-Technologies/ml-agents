using UnityEngine;
using System.Text.RegularExpressions;

namespace MLAgents
{
    /// <summary>
    /// Demonstration Recorder Component.
    /// </summary>
    [RequireComponent(typeof(Agent))]
    public class DemonstrationRecorder : MonoBehaviour
    {
        public bool record;
        public string demonstrationName;
        private Agent m_RecordingAgent;
        private string m_FilePath;
        private DemonstrationStore m_DemoStore;
        public const int MaxNameLength = 16;

        private void Start()
        {
            if (Application.isEditor && record)
            {
                InitializeDemoStore();
            }
        }

        private void Update()
        {
            if (Application.isEditor && record && m_DemoStore == null)
            {
                InitializeDemoStore();
            }
        }

        /// <summary>
        /// Creates demonstration store for use in recording.
        /// </summary>
        private void InitializeDemoStore()
        {
            m_RecordingAgent = GetComponent<Agent>();
            m_DemoStore = new DemonstrationStore();
            demonstrationName = SanitizeName(demonstrationName, MaxNameLength);
            m_DemoStore.Initialize(
                demonstrationName,
                GetComponent<BehaviorParameters>().brainParameters,
                GetComponent<BehaviorParameters>().behaviorName);
            Monitor.Log("Recording Demonstration of Agent: ", m_RecordingAgent.name);
        }

        /// <summary>
        /// Removes all characters except alphanumerics from demonstration name.
        /// Shorten name if it is longer than the maxNameLength.
        /// </summary>
        public static string SanitizeName(string demoName, int maxNameLength)
        {
            var rgx = new Regex("[^a-zA-Z0-9 -]");
            demoName = rgx.Replace(demoName, "");
            // If the string is too long, it will overflow the metadata.
            if (demoName.Length > maxNameLength)
            {
                demoName = demoName.Substring(0, maxNameLength);
            }
            return demoName;
        }

        /// <summary>
        /// Forwards AgentInfo to Demonstration Store.
        /// </summary>
        public void WriteExperience(AgentInfo info)
        {
            m_DemoStore.Record(info);
        }

        /// <summary>
        /// Closes Demonstration store.
        /// </summary>
        private void OnApplicationQuit()
        {
            if (Application.isEditor && record && m_DemoStore != null)
            {
                m_DemoStore.Close();
            }
        }
    }
}
