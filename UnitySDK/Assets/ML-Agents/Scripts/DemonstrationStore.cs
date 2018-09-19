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
    }
    
    public class DemonstrationStore
    {
        private string filePath;
        private DemonstrationMetaData metaData;
        
        public void CreateDirectory()
        {
            if (!Directory.Exists("Assets/Demonstrations"))
            {
                Directory.CreateDirectory("Assets/Demonstrations");
            }
        }
        
        /// <summary>
        /// Creates demonstration file, and writes brainParameters as json to file.
        /// </summary>
        public void CreateDemonstrationFile(string demonstrationName)
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
            
            metaData = new DemonstrationMetaData
            {
                numberEpisodes = 0,
                numberExperiences = 0
            };
        }

        /// <summary>
        /// Writes brain parameters as json to file.
        /// </summary>
        public void WriteBrainParameters(BrainParameters brainParameters)
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
        /// Writes meta-data as json to file.
        /// </summary>
        public void WriteMetadata()
        {
            // Write meta-data to file.
            var jsonInfo = JsonUtility.ToJson(metaData);
            var writer = new StreamWriter(filePath, true);
            writer.WriteLine(jsonInfo);
            writer.Close();
        }
    }
}
