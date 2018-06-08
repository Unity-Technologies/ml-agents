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
        [MenuItem("ML-Agents/Create Scriptable Brain")]
        static public ScriptableBrain CreateScriptableBrain()
        {
            return InstantiateArchetype<ScriptableBrain>();
        }

        [MenuItem("ML-Agents/Create Human Brain")]
        static public HumanBrain CreateHumanBrain()
        {
            return InstantiateArchetype<HumanBrain>();
        }
        
        static public T InstantiateArchetype<T>() where T : ScriptableObject 
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

            string assetPathAndName = AssetDatabase.GenerateUniqueAssetPath(path + "/New " + typeof(T).ToString() + ".asset");

            AssetDatabase.CreateAsset(asset, assetPathAndName);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            EditorUtility.FocusProjectWindow();
            Selection.activeObject = asset;
            return asset;
        }
    }
}