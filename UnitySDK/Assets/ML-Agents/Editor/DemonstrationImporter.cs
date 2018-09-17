using System;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEditor.Experimental.AssetImporters;

namespace MLAgents
{
    [ScriptedImporter(1, new[] {"demo"})]
    public class DemonstrationImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext ctx)
        {
            string demonstrationName = Path.GetFileName(ctx.assetPath);

            var inputType = Path.GetExtension(ctx.assetPath);
            if (inputType == null)
            {
                throw new Exception("Demonstration import error.");
            }

            var demonstration = ScriptableObject.CreateInstance<Demonstration>();
            demonstration.demonstrationName = demonstrationName;
            userData = demonstration.ToString();

            var allLines = File.ReadLines(ctx.assetPath);
            var enumerable = allLines.ToList();
            var brainParams = JsonUtility.FromJson<BrainParameters>(enumerable.First());
            var metaData = JsonUtility.FromJson<DemonstrationMetaData>(enumerable.Last());
            demonstration.brainParameters = brainParams;
            demonstration.metaData = metaData;
            

#if UNITY_2017_3_OR_NEWER
            ctx.AddObjectToAsset(ctx.assetPath, demonstration);
            ctx.SetMainObject(demonstration);
#else
            ctx.SetMainAsset(ctx.assetPath, model);
#endif
        }
    }
}
