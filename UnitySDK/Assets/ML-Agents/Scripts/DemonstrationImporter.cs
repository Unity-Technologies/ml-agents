using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor.Experimental.AssetImporters;

namespace MLAgents
{
	[ScriptedImporter(1, new [] {"demo"} )]
	public class DemonstrationImporter : ScriptedImporter
	{
		public override void OnImportAsset(AssetImportContext ctx)
		{
			string demonstrationName = Path.GetFileName(ctx.assetPath);
			Debug.Log("Importing demonstration " + demonstrationName);

			var inputType = Path.GetExtension(ctx.assetPath);
			if (inputType == null)
			{
				throw new Exception("Demonstration import error.");
			}

			var demonstration = ScriptableObject.CreateInstance<Demonstration>();
			demonstration.demonstrationName = demonstrationName;
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
