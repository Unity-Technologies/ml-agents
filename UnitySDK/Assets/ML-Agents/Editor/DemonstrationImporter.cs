using System;
using System.IO;
using System.Linq;
using MLAgents.CommunicatorObjects;
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
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var inputType = Path.GetExtension(ctx.assetPath);
            if (inputType == null)
            {
                throw new Exception("Demonstration import error.");
            }
            
            // Read first two proto objects containing metadata and brain parameters.
            Stream reader = File.OpenRead(ctx.assetPath);
            
            var metaDataProto = DemonstrationMetaProto.Parser.ParseDelimitedFrom(reader);
            var metaData = new DemonstrationMetaData(metaDataProto);

            reader.Seek(DemonstrationStore.InitialLength + 1, 0);
            var brainParamsProto = BrainParametersProto.Parser.ParseDelimitedFrom(reader);
            var brainParameters = new BrainParameters(brainParamsProto);
            
            reader.Close();
            
            var demonstration = ScriptableObject.CreateInstance<Demonstration>();
            demonstration.Initialize(brainParameters, metaData);
            userData = demonstration.ToString();

#if UNITY_2017_3_OR_NEWER
            ctx.AddObjectToAsset(ctx.assetPath, demonstration);
            ctx.SetMainObject(demonstration);
#else
            ctx.SetMainAsset(ctx.assetPath, model);
#endif
        }
    }
}
