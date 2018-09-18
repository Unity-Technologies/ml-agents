using UnityEngine;
using System.IO;

namespace MLAgents
{
    /// <summary>
    /// Demonstration meta-data.
    /// </summary>
    [System.Serializable]
    public struct DemonstrationMetaData
    {
        public int numberExperiences;
        public int numberEpisodes;
    }

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
        private DemonstrationMetaData metaData;

        /// <summary>
        /// Creates demonstraion file if in Editor mode, and set to record.
        /// </summary>
        private void Start()
        {
            if (Application.isEditor && record)
            {
                recordingAgent = GetComponent<Agent>();
                if (!Directory.Exists("Assets/Demonstrations"))
                {
                    Directory.CreateDirectory("Assets/Demonstrations");
                }

                metaData = new DemonstrationMetaData
                {
                    numberEpisodes = 0,
                    numberExperiences = 0
                };
                CreateDemonstrationFile();
            }
        }

        /// <summary>
        /// Creates demonstration file, and writes brainParameters as json to file.
        /// </summary>
        private void CreateDemonstrationFile()
        {
            // Creates demonstration file.
            var literalName = demonstrationName;
            filePath = "Assets/Demonstrations/" + literalName + ".demo";
            var uniqueNameCounter = 0;
            while (File.Exists(filePath))
            {
                literalName = demonstrationName + "_" + uniqueNameCounter;
                filePath = "Assets/Demonstrations/" + literalName + ".demo";
                uniqueNameCounter++;
            }

            // Writes BrainParameters to file.
            var jsonParameters = JsonUtility.ToJson(recordingAgent.brain.brainParameters);
            var writer = File.CreateText(filePath);
            writer.Write(jsonParameters + '\n');
            writer.Close();
        }

        /// <summary>
        /// Write AgentInfo experience to file as json.
        /// </summary>
        /// <param name="info">AgentInfo of current experience</param>
        public void WriteExperience(AgentInfo info)
        {
            // Increment meta-data counters.
            metaData.numberExperiences++;
            if (info.done)
            {
                metaData.numberEpisodes++;
            }

            // Write AgentInfo to file.
            var jsonInfo = JsonUtility.ToJson(info);
            var writer = new StreamWriter(filePath, true);
            writer.WriteLine(jsonInfo);
            writer.Close();
        }

        /// <summary>
        /// Adds meta-data to demonstration file on exit.
        /// </summary>
        private void OnApplicationQuit()
        {
            if (Application.isEditor && record)
            {
                // Write meta-data to file.
                var jsonInfo = JsonUtility.ToJson(metaData);
                var writer = new StreamWriter(filePath, true);
                writer.WriteLine(jsonInfo);
                writer.Close();
            }
        }
    }
}
