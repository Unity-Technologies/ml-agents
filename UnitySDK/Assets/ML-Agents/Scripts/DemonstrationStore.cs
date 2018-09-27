using System.IO;
using Google.Protobuf;
using MLAgents.CommunicatorObjects;
using UnityEditor;

namespace MLAgents
{
    
    public class DemonstrationStore
    {
        private string filePath;
        private DemonstrationMetaData metaData;
        private const string DemoDirecory = "Assets/Demonstrations/";
        private Stream writer;
        private BrainParameters cachedBrainParameters;
        private float cumulativeReward;
        public const int InitialLength = 32;

        /// <summary>
        /// Initializes the Demonstration Store, and writes initial data.
        /// </summary>
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
            metaData = new DemonstrationMetaData {demonstrationName = demonstrationName};
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
            cumulativeReward += info.reward;
            if (info.done)
            {
                EndEpisode();
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
            EndEpisode();
            metaData.meanReward = cumulativeReward / metaData.numberEpisodes;
            WriteMetadata();
        }

        /// <summary>
        /// Peroforms necessary episode-completion steps.
        /// </summary>
        private void EndEpisode()
        {
            metaData.numberEpisodes += 1;
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
            AssetDatabase.Refresh();
        }
    }
}
