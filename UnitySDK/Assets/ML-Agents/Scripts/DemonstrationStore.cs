using System;
using System.Globalization;
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
    public class DemonstrationMetaData
    {
        public int numberExperiences;
        public int numberEpisodes;
        public float meanReward;
        public const int ApiVersion = 1;

        public DemonstrationMetaData()
        {
        }

        public DemonstrationMetaData(DemonstrationMetaProto demoProto)
        {
            numberEpisodes = demoProto.NumberEpisodes;
            numberExperiences = demoProto.NumberSteps;
            meanReward = demoProto.MeanReward;
            if (demoProto.ApiVersion != ApiVersion)
            {
                throw new Exception("API versions of demonstration are incompatible.");
            }
        }

        public DemonstrationMetaProto ToProto()
        {
            var demoProto = new DemonstrationMetaProto
            {
                ApiVersion = ApiVersion,
                MeanReward = meanReward,
                NumberSteps = numberExperiences,
                NumberEpisodes = numberEpisodes
            };
            return demoProto;
        }
    }

    public class DemonstrationStore
    {
        private string filePath;
        private DemonstrationMetaData metaData;
        private const string DemoDirecory = "Assets/Demonstrations/";
        private Stream writer;
        private BrainParameters cachedBrainParameters;
        public const int InitialLength = 20;

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
            var metaProto = metaData.ToProto();
            metaProto.WriteDelimitedTo(writer);
        }

        /// <summary>
        /// Writes brain parameters to file.
        /// </summary>
        private void WriteBrainParameters(string brainName)
        {
            // Writes BrainParameters to file.
            writer.Seek(InitialLength + 1, 0);
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
            var metaProto = metaData.ToProto();
            var metaProtoBytes = metaProto.ToByteArray();
            writer.Write(metaProtoBytes, 0, metaProtoBytes.Length);
            writer.Seek(0, 0);
            metaProto.WriteDelimitedTo(writer);
            writer.Close();
        }
    }
}
