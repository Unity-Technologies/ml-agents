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
        private Agent recordingAgent;
        private string filePath;
        private DemonstrationStore demoStore;
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
            if (Application.isEditor && record && demoStore == null)
            {
                InitializeDemoStore();
            }
        }

        /// <summary>
        /// Creates demonstration store for use in recording.
        /// </summary>
        private void InitializeDemoStore()
        {
            recordingAgent = GetComponent<Agent>();
            demoStore = new DemonstrationStore();
            demonstrationName = SanitizeName(demonstrationName, MaxNameLength);
            demoStore.Initialize(
                demonstrationName, 
                recordingAgent.brain.brainParameters, 
                recordingAgent.brain.name);            
            Monitor.Log("Recording Demonstration of Agent: ", recordingAgent.name);
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
            demoStore.Record(info);
        }

        /// <summary>
        /// Closes Demonstration store.
        /// </summary>
        private void OnApplicationQuit()
        {
            if (Application.isEditor && record && demoStore != null)
            {
                demoStore.Close();
            }
        }
    }
}
