using UnityEditor;
using UnityEngine;
using System.Collections;

namespace TMPro.EditorUtilities
{

    //[InitializeOnLoad]
    class TMP_ResourcesLoader
    {

        /// <summary>
        /// Function to pre-load the TMP Resources
        /// </summary>
        public static void LoadTextMeshProResources()
        {
            //TMP_Settings.LoadDefaultSettings();
            //TMP_StyleSheet.LoadDefaultStyleSheet();
        }


        static TMP_ResourcesLoader()
        {
            //Debug.Log("Loading TMP Resources...");

            // Get current targetted platform


            //string Settings = PlayerSettings.GetScriptingDefineSymbolsForGroup(BuildTargetGroup.Standalone);
            //TMPro.TMP_Settings.LoadDefaultSettings();
            //TMPro.TMP_StyleSheet.LoadDefaultStyleSheet();
        }



        //[RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        //static void OnBeforeSceneLoaded()
        //{
            //Debug.Log("Before scene is loaded.");

            //    //TMPro.TMP_Settings.LoadDefaultSettings();
            //    //TMPro.TMP_StyleSheet.LoadDefaultStyleSheet();

            //    //ShaderVariantCollection collection = new ShaderVariantCollection();
            //    //Shader s0 = Shader.Find("TextMeshPro/Mobile/Distance Field");
            //    //ShaderVariantCollection.ShaderVariant tmp_Variant = new ShaderVariantCollection.ShaderVariant(s0, UnityEngine.Rendering.PassType.Normal, string.Empty);

            //    //collection.Add(tmp_Variant);
            //    //collection.WarmUp();
        //}

    }

    //static class TMP_ProjectSettings
    //{
    //    [InitializeOnLoadMethod]
    //    static void SetProjectDefineSymbols()
    //    {
    //        string currentBuildSettings = PlayerSettings.GetScriptingDefineSymbolsForGroup(EditorUserBuildSettings.selectedBuildTargetGroup);

    //        //Check for and inject TMP_INSTALLED
    //        if (!currentBuildSettings.Contains("TMP_PRESENT"))
    //        {
    //            PlayerSettings.SetScriptingDefineSymbolsForGroup(EditorUserBuildSettings.selectedBuildTargetGroup, currentBuildSettings + ";TMP_PRESENT");
    //        }
    //    }
    //}
}
