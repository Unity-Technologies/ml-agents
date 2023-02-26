using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using Unity.MLAgents.CommunicatorObjects;
using UnityEngine;
using Random = UnityEngine.Random;


namespace Unity.MLAgents.Demonstrations
{
    public class DemonstrationReader : MonoBehaviour
    {
        public string DemonstrationDirectory;
        string[] m_DemoFileList;
        int m_NumFiles;

        void Awake()
        {
            if (DemonstrationDirectory == "")
            {
                throw new DirectoryNotFoundException("Demonstration Directory must be specified.");
            }
            m_DemoFileList = Directory.GetFiles(DemonstrationDirectory, "*.demo");
            m_NumFiles = m_DemoFileList.Length;
        }

        public float[] GetRandomInitState()
        {
            var fileIndex = Random.Range(0, m_NumFiles);
            Stream reader = File.OpenRead(m_DemoFileList[fileIndex]);

            // Read first three proto objects containing metadata, brain parameters, and observations.

            var metaDataProto = DemonstrationMetaProto.Parser.ParseDelimitedFrom(reader);
            var metaData = metaDataProto.ToDemonstrationMetaData();

            reader.Seek(DemonstrationWriter.MetaDataBytes + 1, 0);
            var brainParamsProto = BrainParametersProto.Parser.ParseDelimitedFrom(reader);
            var brainParameters = brainParamsProto.ToBrainParameters();

            // Read the first AgentInfoActionPair so that we can get the observation sizes.
            List<ObservationSummary> observationSummaries;
            AgentInfoActionPairProto agentInfoActionPairProto;
            try
            {
                agentInfoActionPairProto = AgentInfoActionPairProto.Parser.ParseDelimitedFrom(reader);
                observationSummaries = agentInfoActionPairProto.GetObservationSummaries();
            }
            catch
            {
                throw new Exception("Empty demo file!");

            }

            // for (int i = 1; i < metaDataProto.NumberSteps; i++)
            // {
            //     agentInfoActionPairProto = AgentInfoActionPairProto.Parser.ParseDelimitedFrom(reader);
            //     observationSummaries = agentInfoActionPairProto.GetObservationSummaries();
            // }

            var agentInfoActionPairBytes = agentInfoActionPairProto.CalculateSize() + 2;

            var agentInfoIndex = Random.Range(0, metaDataProto.NumberSteps - 1);

            if (agentInfoIndex == 0)
            {
                //return agentInfoIndex
            }
            else
            {
                try
                {
                    reader.Seek(agentInfoActionPairBytes * agentInfoIndex, SeekOrigin.Current);
                    agentInfoActionPairProto = AgentInfoActionPairProto.Parser.ParseDelimitedFrom(reader);
                    observationSummaries = agentInfoActionPairProto.GetObservationSummaries();
                }
                catch
                {
                    throw new Exception($"Empty agent info action pair in {metaDataProto.DemonstrationName} at position {agentInfoIndex}!");
                }
            }

            reader.Close();

            var observationList = new List<float[]>();
            var observations = agentInfoActionPairProto.AgentInfo.Observations;
            return observations[0].FloatData.Data.ToArray();
        }

        // void Update()
        // {
        //     GetRandomInitState();
        // }
    }
}
