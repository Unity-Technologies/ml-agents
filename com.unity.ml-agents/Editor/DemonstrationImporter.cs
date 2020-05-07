using System;
using System.Collections.Generic;
using System.IO;
using Unity.MLAgents.CommunicatorObjects;
using UnityEditor;
using UnityEngine;
using UnityEditor.Experimental.AssetImporters;
using Unity.MLAgents.Demonstrations;

namespace Unity.MLAgents.Editor
{
    /// <summary>
    /// Asset Importer used to parse demonstration files.
    /// </summary>
    [ScriptedImporter(1, new[] {"demo"})]
    internal class DemonstrationImporter : ScriptedImporter
    {
        const string k_IconPath = "Packages/com.unity.ml-agents/Editor/Icons/DemoIcon.png";

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var inputType = Path.GetExtension(ctx.assetPath);
            if (inputType == null)
            {
                throw new Exception("Demonstration import error.");
            }

            try
            {
                // Read first three proto objects containing metadata, brain parameters, and observations.
                Stream reader = File.OpenRead(ctx.assetPath);

                var metaDataProto = DemonstrationMetaProto.Parser.ParseDelimitedFrom(reader);
                var metaData = metaDataProto.ToDemonstrationMetaData();

                reader.Seek(DemonstrationWriter.MetaDataBytes + 1, 0);
                var brainParamsProto = BrainParametersProto.Parser.ParseDelimitedFrom(reader);
                var brainParameters = brainParamsProto.ToBrainParameters();

                // Read the first AgentInfoActionPair so that we can get the observation sizes.
                List<ObservationSummary> observationSummaries;
                try
                {
                    var agentInfoActionPairProto = AgentInfoActionPairProto.Parser.ParseDelimitedFrom(reader);
                    observationSummaries = agentInfoActionPairProto.GetObservationSummaries();
                }
                catch
                {
                    // Just in case there weren't any AgentInfoActionPair or they couldn't be read.
                    observationSummaries = new List<ObservationSummary>();
                }

                reader.Close();

                var demonstrationSummary = ScriptableObject.CreateInstance<DemonstrationSummary>();
                demonstrationSummary.Initialize(brainParameters, metaData, observationSummaries);
                userData = demonstrationSummary.ToString();

                var texture = (Texture2D)
                    AssetDatabase.LoadAssetAtPath(k_IconPath, typeof(Texture2D));

                ctx.AddObjectToAsset(ctx.assetPath, demonstrationSummary, texture);
                ctx.SetMainObject(demonstrationSummary);
            }
            catch
            {
                // ignored
            }
        }
    }
}
