using UnityEngine;
using System.IO;

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

        /// <summary>
        /// Initializes Demonstration store.
        /// </summary>
        private void Start()
        {
            if (Application.isEditor && record)
            {
                recordingAgent = GetComponent<Agent>();
                demoStore = new DemonstrationStore();
                demoStore.Initialize(demonstrationName, recordingAgent.brain.brainParameters);               
            }
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
