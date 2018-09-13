using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
using MLAgents;

namespace Unity.MLAgents
{
    public class BrainInstantiation
    {
        [MenuItem("ML-Agents/Create Brain/Heuristic")]
        public static HeuristicBrain CreateScriptableBrain()
        {
            return InstantiateArchetype<HeuristicBrain>("HeuristicBrain");
        }

        [MenuItem("ML-Agents/Create Brain/Player")]
        public static PlayerBrain CreateHumanBrain()
        {
            return InstantiateArchetype<PlayerBrain>("PlayerBrain");
        }
        
        [MenuItem("ML-Agents/Create Brain/Internal")]
        public static InternalBrain CreateLearnedBrain()
        {
            return InstantiateArchetype<InternalBrain>("InternalBrain");
        }
        
        public static T InstantiateArchetype<T>(string name) where T : ScriptableObject 
        {
            T asset = ScriptableObject.CreateInstance<T>();

            string path = AssetDatabase.GetAssetPath(Selection.activeObject);
            if (path == "")
            {
                path = "Assets";
            }
            else if (Path.GetExtension(path) != "")
            {
                path = path.Replace(
                    Path.GetFileName(AssetDatabase.GetAssetPath(Selection.activeObject)), "");
            }

            string assetPathAndName = AssetDatabase.GenerateUniqueAssetPath(
                path + "/New" + name + ".asset");

            AssetDatabase.CreateAsset(asset, assetPathAndName);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            EditorUtility.FocusProjectWindow();
            Selection.activeObject = asset;
            return asset;
        }
    }
}