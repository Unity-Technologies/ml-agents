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
        /// Creates demonstraion file if in Editor mode, and set to record.
        /// </summary>
        private void Start()
        {
            if (Application.isEditor && record)
            {
                recordingAgent = GetComponent<Agent>();
                demoStore = new DemonstrationStore();
                demoStore.CreateDirectory();                

                demoStore.CreateDemonstrationFile(demonstrationName);
                demoStore.WriteBrainParameters(recordingAgent.brain.brainParameters);
            }
        }

        /// <summary>
        /// Forwards AgentInfo to Demonstration Store.
        /// </summary>
        public void WriteExperience(AgentInfo info)
        {
            demoStore.WriteExperience(info);
        }

        /// <summary>
        /// Adds meta-data to demonstration file on exit.
        /// </summary>
        private void OnApplicationQuit()
        {
            if (Application.isEditor && record)
            {
                demoStore.WriteMetadata();
            }
        }
    }
}
