using System;
using System.Collections.Generic;
using UnityEngine;
using Barracuda;
using System.IO;
using MLAgents;

namespace MLAgentsExamples
{
    /// <summary>
    /// Utility class to allow the NNModel file for an agent to be overriden during inference.
    /// This is used internally to validate the file after training is done.
    /// The behavior name to override and file path are specified on the commandline, e.g.
    /// player.exe --mlagents-override-model behavior1 /path/to/model1.nn --mlagents-override-model behavior2 /path/to/model2.nn
    ///
    /// Additionally, a number of episodes to run can be specified; after this, the application will quit.
    /// Note this will only work with example scenes that have 1:1 Agent:Behaviors. More complicated scenes like WallJump
    /// probably won't override correctly.
    /// </summary>
    public class ModelOverrider : MonoBehaviour
    {
        const string k_CommandLineModelOverrideFlag = "--mlagents-override-model";
        const string k_CommandLineQuitAfterEpisodesFlag = "--mlagents-quit-after-episodes";

        // The attached Agent
        Agent m_Agent;

        // Assets paths to use, with the behavior name as the key.
        Dictionary<string, string> m_BehaviorNameOverrides = new Dictionary<string, string>();

        // Cached loaded NNModels, with the behavior name as the key.
        Dictionary<string, NNModel> m_CachedModels = new Dictionary<string, NNModel>();

        // Max episodes to run. Only used if > 0
        // Will default to 1 if override models are specified, otherwise 0.
        int m_MaxEpisodes;

        int m_NumSteps;

        /// <summary>
        /// Get the asset path to use from the commandline arguments.
        /// </summary>
        /// <returns></returns>
        void GetAssetPathFromCommandLine()
        {
            m_BehaviorNameOverrides.Clear();

            var maxEpisodes = 0;

            var args = Environment.GetCommandLineArgs();
            for (var i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == k_CommandLineModelOverrideFlag && i < args.Length-2)
                {
                    var key = args[i + 1].Trim();
                    var value = args[i + 2].Trim();
                    m_BehaviorNameOverrides[key] = value;
                }
                else if (args[i] == k_CommandLineQuitAfterEpisodesFlag)
                {
                    Int32.TryParse(args[i + 1], out maxEpisodes);
                }
            }

            if (m_BehaviorNameOverrides.Count > 0)
            {
                // If overriding models, set maxEpisodes to 1 or the command line value
                m_MaxEpisodes = maxEpisodes > 0 ? maxEpisodes : 1;
                Debug.Log($"setting m_MaxEpisodes to {maxEpisodes}");
            }
        }

        void OnEnable()
        {
            m_Agent = GetComponent<Agent>();

            GetAssetPathFromCommandLine();
            if (m_BehaviorNameOverrides.Count > 0)
            {
                OverrideModel();
            }
        }

        void FixedUpdate()
        {
            if (m_MaxEpisodes > 0)
            {
                if (m_NumSteps > m_MaxEpisodes * m_Agent.maxStep)
                {
                    // Stop recording so that we don't write partial rewards to the timer info.
                    TimerStack.Instance.Recording = false;
                    Application.Quit(0);
                }
            }
            m_NumSteps++;
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
            catch(IOException)
            {
                Debug.Log($"Couldn't load file {assetPath}", this);
                // Cache the null so we don't repeatedly try to load a missing file
                m_CachedModels[behaviorName] = null;
                return null;
            }

            var asset = ScriptableObject.CreateInstance<NNModel>();
            asset.modelData = ScriptableObject.CreateInstance<NNModelData>();
            asset.modelData.Value = model;

            asset.name = "Override - " + Path.GetFileName(assetPath);
            m_CachedModels[behaviorName] = asset;
            return asset;
        }

        /// <summary>
        /// Load the NNModel file from the specified path, and give it to the attached agent.
        /// </summary>
        void OverrideModel()
        {
            m_Agent.LazyInitialize();
            var bp = m_Agent.GetComponent<BehaviorParameters>();

            var nnModel = GetModelForBehaviorName(bp.behaviorName);
            Debug.Log($"Overriding behavior {bp.behaviorName} for agent with model {nnModel?.name}");
            // This might give a null model; that's better because we'll fall back to the Heuristic
            m_Agent.GiveModel($"Override_{bp.behaviorName}", nnModel);

        }
    }
}
