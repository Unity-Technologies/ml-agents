using System.IO;
using UnityEngine;


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
        public const int API_VERSION = 1;
    }
    
    public class DemonstrationStore
    {
        private string filePath;
        private DemonstrationMetaData metaData;
        private const string DemoDirecory = "Assets/Demonstrations";

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
        /// Creates demonstration file, and writes brainParameters as json to file.
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
            
            metaData = new DemonstrationMetaData
            {
                numberEpisodes = 0,
                numberExperiences = 0,
            };
        }

        /// <summary>
        /// Writes brain parameters as json to file.
        /// </summary>
        private void WriteBrainParameters(BrainParameters brainParameters)
        {
            // Writes BrainParameters to file.
            var jsonParameters = JsonUtility.ToJson(brainParameters);
            var writer = File.CreateText(filePath);
            writer.Write(jsonParameters + '\n');
            writer.Close();
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
            var writer = new StreamWriter(filePath, true);
            writer.WriteLine(jsonInfo);
            writer.Close();
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
            var writer = new StreamWriter(filePath, true);
            writer.WriteLine(jsonInfo);
            writer.Close();
        }
    }
}
