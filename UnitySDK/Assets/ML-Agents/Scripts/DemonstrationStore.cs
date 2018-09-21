using System.IO;
using System.Linq;
using Google.Protobuf;
using MLAgents.CommunicatorObjects;
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
        private Stream writer;
        private BrainParameters cachedBrainParameters;

        public void Initialize(string demonstrationName, BrainParameters brainParameters, string brainName)
        {
            cachedBrainParameters = brainParameters;
            CreateDirectory();
            CreateDemonstrationFile(demonstrationName);
            WriteBrainParameters(brainName);
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

            writer = File.Create(filePath);
            metaData = new DemonstrationMetaData();
        }

        /// <summary>
        /// Writes brain parameters to file.
        /// </summary>
        private void WriteBrainParameters(string brainName)
        {
            // Writes BrainParameters to file.
            var brainProto = cachedBrainParameters.ToProto(brainName, BrainTypeProto.Player);
            brainProto.WriteDelimitedTo(writer);
        }

        /// <summary>
        /// Write AgentInfo experience to file.
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
            var agentProto = info.ToProto();
            agentProto.WriteDelimitedTo(writer);
            
        }

        /// <summary>
        /// Performs all clean-up necessary
        /// </summary>
        public void Close()
        {
            WriteMetadata();
        }

        /// <summary>
        /// Writes meta-data and closes session.
        /// </summary>
        private void WriteMetadata()
        {
            // Todo re-implement meta-data
            writer.Close();
        }
    }
}
