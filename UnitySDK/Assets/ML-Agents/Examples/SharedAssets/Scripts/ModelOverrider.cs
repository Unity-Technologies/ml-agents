using System;
using UnityEngine;
using Barracuda;
using System.IO;
namespace MLAgents
{
    /// <summary>
    /// Utility class to allow the NNModel file for an agent to be overriden during inference.
    /// This is useful to validate the file after training is done.
    /// </summary>
    public class ModelOverrider : MonoBehaviour
    {
        string m_AssetPath;
        const string k_CommandLineFlag = "--mlagents-override-model";
        
        void OnEnable()
        {
            m_AssetPath = GetAssetPathFromCommandLine();
            if (m_AssetPath != null)
            {
                OverrideModel();
            }
        }

        /// <summary>
        /// Get the asset path to use from the commandline arguments.
        /// </summary>
        /// <returns></returns>
        static string GetAssetPathFromCommandLine()
        {
            var args = Environment.GetCommandLineArgs();
            for (var i = 0; i < args.Length; i++)
            {
                if (args[i] == k_CommandLineFlag)
                {
                    return args[i + 1];
                }
            }

            return null;
        }
        
        /// <summary>
        /// Load the NNModel file from the specified path, and give it to the attached agent.
        /// </summary>
        void OverrideModel()
        {
            var model = File.ReadAllBytes(m_AssetPath);
            var asset = ScriptableObject.CreateInstance<NNModel>();
            asset.Value = model;
            asset.name = "Override - " + Path.GetFileName(m_AssetPath);
            
            var agent = GetComponent<Agent>();
            Debug.Log("Overriding the asset for agent...");
            agent.GiveModel("trainedModel", asset, InferenceDevice.CPU);
        }
    }
}
