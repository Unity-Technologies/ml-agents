using System;
using System.IO;
using MLAgents.CommunicatorObjects;
using UnityEditor;
using UnityEngine;
using UnityEditor.Experimental.AssetImporters;

namespace MLAgents
{
    /// <summary>
    /// Asset Importer used to parse demonstration files.
    /// </summary>
    [ScriptedImporter(1, new[] {"demo"})]
    public class DemonstrationImporter : ScriptedImporter
    {
        private const string k_IconPath = "Assets/ML-Agents/Resources/DemoIcon.png";

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

                reader.Seek(DemonstrationStore.MetaDataBytes + 1, 0);
                var brainParamsProto = BrainParametersProto.Parser.ParseDelimitedFrom(reader);
                var brainParameters = brainParamsProto.ToBrainParameters();

                reader.Close();

                var demonstration = ScriptableObject.CreateInstance<Demonstration>();
                demonstration.Initialize(brainParameters, metaData);
                userData = demonstration.ToString();

                var texture = (Texture2D)
                    AssetDatabase.LoadAssetAtPath(k_IconPath, typeof(Texture2D));

#if UNITY_2017_3_OR_NEWER
                ctx.AddObjectToAsset(ctx.assetPath, demonstration, texture);
                ctx.SetMainObject(demonstration);
#else
                ctx.SetMainAsset(ctx.assetPath, demonstration);
#endif
            }
            catch
            {
                // ignored
            }
        }
    }
}
