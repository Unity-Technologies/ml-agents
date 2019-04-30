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
        public const int MAX_NAME_LENGTH = 16;

        /// <summary>
        /// Initializes Demonstration store.
        /// </summary>
        private void Start()
        {
            if (Application.isEditor && record)
            {
                recordingAgent = GetComponent<Agent>();
                demoStore = new DemonstrationStore();
                demonstrationName = SanitizeName(demonstrationName, MAX_NAME_LENGTH);
                demoStore.Initialize(
                    demonstrationName, 
                    recordingAgent.brain.brainParameters, 
                    recordingAgent.brain.name);            
                Monitor.Log("Recording Demonstration of Agent: ", recordingAgent.name);
            }
        }

        /// <summary>
        /// Removes all characters except alphanumerics from demonstration name.
        /// Shorten name if it is too long for the metadata header.
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
            if (Application.isEditor && record)
            {
                demoStore.Close();
            }
        }
    }
}
