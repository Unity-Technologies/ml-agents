using System.IO;
using Google.Protobuf;
using System.Collections.Generic;

namespace MLAgents
{
    /// <summary>
    /// Responsible for writing demonstration data to stream (usually a file stream).
    /// </summary>
    public class DemonstrationWriter
    {
        public const int MetaDataBytes = 32; // Number of bytes allocated to metadata in demo file.

        DemonstrationMetaData m_MetaData;
        Stream m_Writer;
        float m_CumulativeReward;
        WriteAdapter m_WriteAdapter = new WriteAdapter();

        /// <summary>
        /// Create a DemonstrationWriter that will write to the specified stream.
        /// The stream must support writes and seeking.
        /// </summary>
        /// <param name="stream"></param>
        public DemonstrationWriter(Stream stream)
        {
            m_Writer = stream;
        }

        /// <summary>
        /// Writes the initial data to the stream.
        /// </summary>
        public void Initialize(
            string demonstrationName, BrainParameters brainParameters, string brainName)
        {
            if (m_Writer == null)
            {
                // Already closed
                return;
            }

            m_MetaData = new DemonstrationMetaData { demonstrationName = demonstrationName };
            var metaProto = m_MetaData.ToProto();
            metaProto.WriteDelimitedTo(m_Writer);

            WriteBrainParameters(brainName, brainParameters);
        }

        /// <summary>
        /// Writes meta-data. Note that this is called at the *end* of recording, but writes to the
        /// beginning of the file.
        /// </summary>
        void WriteMetadata()
        {
            if (m_Writer == null)
            {
                // Already closed
                return;
            }

            var metaProto = m_MetaData.ToProto();
            var metaProtoBytes = metaProto.ToByteArray();
            m_Writer.Write(metaProtoBytes, 0, metaProtoBytes.Length);
            m_Writer.Seek(0, 0);
            metaProto.WriteDelimitedTo(m_Writer);
        }

        /// <summary>
        /// Writes brain parameters to file.
        /// </summary>
        void WriteBrainParameters(string brainName, BrainParameters brainParameters)
        {
            if (m_Writer == null)
            {
                // Already closed
                return;
            }

            // Writes BrainParameters to file.
            m_Writer.Seek(MetaDataBytes + 1, 0);
            var brainProto = brainParameters.ToProto(brainName, false);
            brainProto.WriteDelimitedTo(m_Writer);
        }

        /// <summary>
        /// Write AgentInfo experience to file.
        /// </summary>
        internal void Record(AgentInfo info, List<ISensor> sensors)
        {
            if (m_Writer == null)
            {
                // Already closed
                return;
            }

            // Increment meta-data counters.
            m_MetaData.numberExperiences++;
            m_CumulativeReward += info.reward;
            if (info.done)
            {
                EndEpisode();
            }

            // Generate observations and add AgentInfo to file.
            var agentProto = info.ToInfoActionPairProto();
            foreach (var sensor in sensors)
            {
                agentProto.AgentInfo.Observations.Add(sensor.GetObservationProto(m_WriteAdapter));
            }

            agentProto.WriteDelimitedTo(m_Writer);
        }


        /// <summary>
        /// Performs all clean-up necessary
        /// </summary>
        public void Close()
        {
            if (m_Writer == null)
            {
                // Already closed
                return;
            }

            EndEpisode();
            m_MetaData.meanReward = m_CumulativeReward / m_MetaData.numberEpisodes;
            WriteMetadata();
            m_Writer.Close();
            m_Writer = null;
        }

        /// <summary>
        /// Performs necessary episode-completion steps.
        /// </summary>
        void EndEpisode()
        {
            m_MetaData.numberEpisodes += 1;
        }
    }
}
