using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
using MLAgents;

namespace Unity.MLAgents
{
    public class Menu
    {
        [MenuItem("ML-Agents/Create Brain/Scriptable")]
        static public ScriptableBrain CreateScriptableBrain()
        {
            return InstantiateArchetype<ScriptableBrain>("ScriptableBrain");
        }

        [MenuItem("ML-Agents/Create Brain/Human")]
        static public HumanBrain CreateHumanBrain()
        {
            return InstantiateArchetype<HumanBrain>("HumanBrain");
        }
        
        [MenuItem("ML-Agents/Create Brain/Learned")]
        static public LearnedBrain CreateLearnedBrain()
        {
            return InstantiateArchetype<LearnedBrain>("LearnedBrain");
        }
        
        static public T InstantiateArchetype<T>(string name) where T : ScriptableObject 
        {
            T asset = ScriptableObject.CreateInstance<T>();

            string path = AssetDatabase.GetAssetPath(Selection.activeObject);
            if (path == "")
            {
                path = "Assets";
            }
            else if (Path.GetExtension(path) != "")
            {
                path = path.Replace(Path.GetFileName(AssetDatabase.GetAssetPath(Selection.activeObject)), "");
            }

            string assetPathAndName = AssetDatabase.GenerateUniqueAssetPath(path + "/New" + name + ".asset");

            AssetDatabase.CreateAsset(asset, assetPathAndName);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            EditorUtility.FocusProjectWindow();
            Selection.activeObject = asset;
            return asset;
        }
    }
}