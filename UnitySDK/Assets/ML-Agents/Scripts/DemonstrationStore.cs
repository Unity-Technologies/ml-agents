using System.IO;
using UnityEngine;

namespace MLAgents
{
    /// <summary>
    /// Demonstration meta-data.
    /// Kept in a struct for easy serialization and deserialization.
    /// </summary>
    [System.Serializable]
    public struct DemonstrationMetaData
    {
        public int numberExperiences;
        public int numberEpisodes;
        public const int API_VERSION = 1;
    }

    public class DemonstrationStore
    {
        private string filePath;
        private DemonstrationMetaData metaData;
        private const string DemoDirecory = "Assets/Demonstrations/";
        private StreamWriter writer;

        public void Initialize(string demonstrationName, BrainParameters brainParameters)
        {
            CreateDirectory();
            CreateDemonstrationFile(demonstrationName);
            WriteBrainParameters(brainParameters);
        }

        /// <summary>
        /// Checks for the existence of the Demonstrations directory
        /// and creates it if it does not exist.
        /// </summary>
        private void CreateDirectory()
        {
            if (!Directory.Exists(DemoDirecory))
            {
                Directory.CreateDirectory(DemoDirecory);
            }
        }

        /// <summary>
        /// Creates demonstration file.
        /// </summary>
        private void CreateDemonstrationFile(string demonstrationName)
        {
            // Creates demonstration file.
            var literalName = demonstrationName;
            filePath = DemoDirecory + literalName + ".demo";
            var uniqueNameCounter = 0;
            while (File.Exists(filePath))
            {
                literalName = demonstrationName + "_" + uniqueNameCounter;
                filePath = DemoDirecory + literalName + ".demo";
                uniqueNameCounter++;
            }

            writer = File.CreateText(filePath);
            metaData = new DemonstrationMetaData();
        }

        /// <summary>
        /// Writes brain parameters as json to file.
        /// </summary>
        private void WriteBrainParameters(BrainParameters brainParameters)
        {
            // Writes BrainParameters to file.
            var jsonParameters = JsonUtility.ToJson(brainParameters);
            writer.Write(jsonParameters + '\n');
        }

        /// <summary>
        /// Write AgentInfo experience to file as json.
        /// </summary>
        public void Record(AgentInfo info)
        {
            // Increment meta-data counters.
            metaData.numberExperiences++;
            if (info.done)
            {
                metaData.numberEpisodes++;
            }

            // Write AgentInfo to file.
            var jsonInfo = JsonUtility.ToJson(info);
            writer.WriteLine(jsonInfo);
        }

        /// <summary>
        /// Performs all clean-up necessary
        /// </summary>
        public void Close()
        {
            WriteMetadata();
        }

        /// <summary>
        /// Writes meta-data as json to file.
        /// </summary>
        private void WriteMetadata()
        {
            // Write meta-data to file.
            var jsonInfo = JsonUtility.ToJson(metaData);
            writer.WriteLine(jsonInfo);
            writer.Close();
        }
    }
}
