using System.IO;
using System.IO.Abstractions;
using Google.Protobuf;
using MLAgents.CommunicatorObjects;

namespace MLAgents
{
    /// <summary>
    /// Responsible for writing demonstration data to file.
    /// </summary>
    public class DemonstrationStore
    {
        public const int MetaDataBytes = 32; // Number of bytes allocated to metadata in demo file.
        private readonly IFileSystem fileSystem;
        private const string DemoDirecory = "Assets/Demonstrations/";
        private const string ExtensionType = ".demo";

        private string filePath;
        private DemonstrationMetaData metaData;
        private Stream writer;
        private float cumulativeReward;

        public DemonstrationStore(IFileSystem fileSystem)
        {
            this.fileSystem = fileSystem;
        }

        public DemonstrationStore()
        {
            fileSystem = new FileSystem();
        }

        /// <summary>
        /// Initializes the Demonstration Store, and writes initial data.
        /// </summary>
        public void Initialize(
            string demonstrationName, BrainParameters brainParameters, string brainName)
        {
            CreateDirectory();
            CreateDemonstrationFile(demonstrationName);
            WriteBrainParameters(brainName, brainParameters);
        }

        /// <summary>
        /// Checks for the existence of the Demonstrations directory
        /// and creates it if it does not exist.
        /// </summary>
        private void CreateDirectory()
        {
            if (!fileSystem.Directory.Exists(DemoDirecory))
            {
                fileSystem.Directory.CreateDirectory(DemoDirecory);
            }
        }

        /// <summary>
        /// Creates demonstration file.
        /// </summary>
        private void CreateDemonstrationFile(string demonstrationName)
        {
            // Creates demonstration file.
            var literalName = demonstrationName;
            filePath = DemoDirecory + literalName + ExtensionType;
            var uniqueNameCounter = 0;
            while (fileSystem.File.Exists(filePath))
            {
                literalName = demonstrationName + "_" + uniqueNameCounter;
                filePath = DemoDirecory + literalName + ExtensionType;
                uniqueNameCounter++;
            }

            writer = fileSystem.File.Create(filePath);
            metaData = new DemonstrationMetaData {demonstrationName = demonstrationName};
            var metaProto = metaData.ToProto();
            metaProto.WriteDelimitedTo(writer);
        }

        /// <summary>
        /// Writes brain parameters to file.
        /// </summary>
        private void WriteBrainParameters(string brainName, BrainParameters brainParameters)
        {
            // Writes BrainParameters to file.
            writer.Seek(MetaDataBytes + 1, 0);
            var brainProto = brainParameters.ToProto(brainName, false);
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
            writer.Close();
        }

        /// <summary>
        /// Performs necessary episode-completion steps.
        /// </summary>
        private void EndEpisode()
        {
            metaData.numberEpisodes += 1;
        }

        /// <summary>
        /// Writes meta-data.
        /// </summary>
        private void WriteMetadata()
        {
            var metaProto = metaData.ToProto();
            var metaProtoBytes = metaProto.ToByteArray();
            writer.Write(metaProtoBytes, 0, metaProtoBytes.Length);
            writer.Seek(0, 0);
            metaProto.WriteDelimitedTo(writer);
        }
    }
}
