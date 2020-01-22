using UnityEditor;
using UnityEngine;
using System.IO;
using System.Collections;



namespace TMPro.EditorUtilities
{

    public static class TMP_ColorGradientAssetMenu
    {
        [MenuItem("Assets/Create/TextMeshPro/Color Gradient", false, 115)]
        public static void CreateColorGradient(MenuCommand context)
        {
            string filePath;

            if (Selection.assetGUIDs.Length == 0)
                filePath = "Assets/New TMP Color Gradient.asset";
            else
                filePath = AssetDatabase.GUIDToAssetPath(Selection.assetGUIDs[0]);

            if (Directory.Exists(filePath))
            {
                filePath += "/New TMP Color Gradient.asset";
            }
            else
            {
                filePath = Path.GetDirectoryName(filePath) + "/New TMP Color Gradient.asset";
            }

            filePath = AssetDatabase.GenerateUniqueAssetPath(filePath);

            // Create new Color Gradient Asset.
            TMP_ColorGradient colorGradient = ScriptableObject.CreateInstance<TMP_ColorGradient>();

            // Create Asset
            AssetDatabase.CreateAsset(colorGradient, filePath);

            //EditorUtility.SetDirty(colorGradient);

            AssetDatabase.SaveAssets();

            AssetDatabase.ImportAsset(AssetDatabase.GetAssetPath(colorGradient));

            EditorUtility.FocusProjectWindow();
            Selection.activeObject = colorGradient;

        }
    }
}
