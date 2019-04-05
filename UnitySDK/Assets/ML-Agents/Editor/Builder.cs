#if UNITY_CLOUD_BUILD

using System;
using System.Collections.Generic;
using System.Linq;
using MLAgents;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using System.IO;

namespace MLAgents
{
	
	public class Builder
	{
//		[MenuItem("ML-Agents/Run PreExport Method")]
		public static void PreExport()
		{
			var scenePath = Environment.GetEnvironmentVariable("SCENE_PATH"); 
			SwitchAllLearningBrainToControlMode();
			PutSceneToBuild(scenePath);
		}

		protected static void PutSceneToBuild(string scenePath)
		{
			List<EditorBuildSettingsScene> scenes = new List<EditorBuildSettingsScene>();

			EditorBuildSettingsScene targetScene = new EditorBuildSettingsScene(scenePath, true);
			
			if (!File.Exists(scenePath))
			{
				throw new Exception("The Specified scenePath " + scenePath + " does not exist");
			}
			
			scenes.Add(targetScene);

			EditorBuildSettings.scenes = scenes.ToArray();
		}
		
//		[MenuItem("ML-Agents/Switch All Learning Brain To Control Mode")]
		protected static void SwitchAllLearningBrainToControlMode()
		{
			string[] scenePaths = Directory.GetFiles("Assets/ML-Agents/Examples/", "*.unity", SearchOption.AllDirectories);

			foreach (string scenePath in scenePaths)
			{
				var curScene = EditorSceneManager.OpenScene(scenePath);
				var aca = SceneAsset.FindObjectOfType<Academy>();
				if (aca != null)
				{
					var learningBrains = aca.broadcastHub.broadcastingBrains.Where(
						x => x != null && x is LearningBrain);
		
					foreach (Brain brain in learningBrains)
					{
						if (!aca.broadcastHub.IsControlled(brain))
						{
							Debug.Log("Switched brain in scene " + scenePath);
							aca.broadcastHub._brainsToControl.Add(brain);
						}
					}
					EditorSceneManager.SaveScene(curScene);
				}
				else
				{
					Debug.Log("scene " + scenePath + " doesn't have a Academy in it");
				}
			}
		}

	}
}

#endif
