using System;
using System.IO;
using MLAgents.CommunicatorObjects;
using UnityEditor;
using UnityEngine;
using UnityEditor.Experimental.AssetImporters;
using MLAgents.Demonstrations;

namespace MLAgents.Editor
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
                // Read first two proto objects containing metadata and brain parameters.
                Stream reader = File.OpenRead(ctx.assetPath);

                var metaDataProto = DemonstrationMetaProto.Parser.ParseDelimitedFrom(reader);
                var metaData = metaDataProto.ToDemonstrationMetaData();

                reader.Seek(DemonstrationWriter.MetaDataBytes + 1, 0);
                var brainParamsProto = BrainParametersProto.Parser.ParseDelimitedFrom(reader);
                var brainParameters = brainParamsProto.ToBrainParameters();

                reader.Close();

                var demonstration = ScriptableObject.CreateInstance<Demonstration>();
                demonstration.Initialize(brainParameters, metaData);
                userData = demonstration.ToString();

                var texture = (Texture2D)
                    AssetDatabase.LoadAssetAtPath(k_IconPath, typeof(Texture2D));

                ctx.AddObjectToAsset(ctx.assetPath, demonstration, texture);
                ctx.SetMainObject(demonstration);
            }
            catch
            {
                // ignored
            }
        }
    }
}
