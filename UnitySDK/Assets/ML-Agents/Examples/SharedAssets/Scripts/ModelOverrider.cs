using System;
using System.Collections.Generic;
using UnityEngine;
using Barracuda;
using System.IO;
using System.Runtime.Remoting;

namespace MLAgents
{
    /// <summary>
    /// Utility class to allow the NNModel file for an agent to be overriden during inference.
    /// This is useful to validate the file after training is done.
    /// The behavior name to override and file path are specified on the commandline, e.g.
    /// player.exe --mlagents-override-model behavior1 /path/to/model1.nn --mlagents-override-model behavior2 /path/to/model2.nn
    /// </summary>
    public class ModelOverrider : MonoBehaviour
    {
        const string k_CommandLineFlag = "--mlagents-override-model";
        // Assets paths to use, with the behavior name as the key.
        Dictionary<string, string> m_BehaviorNameOverrides = new Dictionary<string, string>();

        // Cached loaded NNModels, with the behavior name as the key.
        Dictionary<string, NNModel> m_CachedModels = new Dictionary<string, NNModel>();

        void OnEnable()
        {
            GetAssetPathFromCommandLine();
            if (m_BehaviorNameOverrides.Count > 0)
            {
                OverrideModel();
            }

        }

        /// <summary>
        /// Get the asset path to use from the commandline arguments.
        /// </summary>
        /// <returns></returns>
        void GetAssetPathFromCommandLine()
        {
            m_BehaviorNameOverrides.Clear();
            m_BehaviorNameOverrides["3DBall"] = "/Users/chris.elion/code/ml-agents/models/ppo/3DBall.nn"; // TODO REMOVE ME

            var args = Environment.GetCommandLineArgs();
            for (var i = 0; i < args.Length-2; i++)
            {
                if (args[i] == k_CommandLineFlag)
                {
                    var key = args[i + 1].Trim();
                    var value = args[i + 2].Trim();
                    m_BehaviorNameOverrides[key] = value;
                }
            }
        }

        NNModel GetModelForBehaviorName(string behaviorName)
        {
            if (m_CachedModels.ContainsKey(behaviorName))
            {
                return m_CachedModels[behaviorName];
            }

            if (!m_BehaviorNameOverrides.ContainsKey(behaviorName))
            {
                Debug.Log($"No override for behaviorName {behaviorName}");
                return null;
            }

            var assetPath = m_BehaviorNameOverrides[behaviorName];

            byte[] model = null;
            try
            {
                model = File.ReadAllBytes(assetPath);
            }
            catch(IOException e)
            {
                Debug.Log($"Couldn't load file {assetPath}", this);
                // Cache the null so we don't repeatedly try to load a missing file
                m_CachedModels[behaviorName] = null;
                return null;
            }

            var asset = ScriptableObject.CreateInstance<NNModel>();
            asset.Value = model;
            asset.name = "Override - " + Path.GetFileName(assetPath);
            m_CachedModels[behaviorName] = asset;
            return asset;
        }

        /// <summary>
        /// Load the NNModel file from the specified path, and give it to the attached agent.
        /// </summary>
        void OverrideModel()
        {
            var agent = GetComponent<Agent>();
            agent.LazyInitialize();
            var bp = agent.GetComponent<BehaviorParameters>();

            var behaviorNameAndTeamId = bp.behaviorName;
            var behaviorName = behaviorNameAndTeamId.Split('?')[0];
            var nnModel = GetModelForBehaviorName(behaviorName);
            Debug.Log($"Overriding behavior {behaviorName} for agent with model {nnModel?.name}");
            // This might give a null model; that's better because we'll fall back to the Heuristic
            agent.GiveModel($"Override_{behaviorName}", nnModel, InferenceDevice.CPU);

        }
    }
}
