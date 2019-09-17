using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEditor.Experimental.AssetImporters;

namespace Barracuda
{
    /// <summary>
    /// Asset Importer of barracuda models.
    /// </summary>
    [ScriptedImporter(1, new[] {"nn"})]
    public class NNModelImporter : ScriptedImporter {
        private const string iconName = "NNModelIcon";

        private Texture2D iconTexture;

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var model = File.ReadAllBytes(ctx.assetPath);
            var asset = ScriptableObject.CreateInstance<NNModel>();
            asset.Value = model;
            
            ctx.AddObjectToAsset("main obj", asset, LoadIconTexture());
            ctx.SetMainObject(asset);
        }

        private Texture2D LoadIconTexture()
        {
            if (iconTexture == null)
            {
                string[] allCandidates = AssetDatabase.FindAssets(iconName);

                if (allCandidates.Length > 0)
                {
                    iconTexture = AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(allCandidates[0]), typeof(Texture2D)) as Texture2D;
                }
            }
            return iconTexture;
        }
        
    }
}
