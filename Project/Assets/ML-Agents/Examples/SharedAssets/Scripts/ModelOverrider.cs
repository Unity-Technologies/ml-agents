using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.IO;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Unity.MLAgentsExamples
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
        const string k_CommandLineModelOverrideDirectoryFlag = "--mlagents-override-model-directory";
        const string k_CommandLineQuitAfterEpisodesFlag = "--mlagents-quit-after-episodes";
        const string k_CommandLineQuitOnLoadFailure = "--mlagents-quit-on-load-failure";

        // The attached Agent
        Agent m_Agent;

        // Assets paths to use, with the behavior name as the key.
        Dictionary<string, string> m_BehaviorNameOverrides = new Dictionary<string, string>();

        string m_BehaviorNameOverrideDirectory;

        // Cached loaded NNModels, with the behavior name as the key.
        Dictionary<string, NNModel> m_CachedModels = new Dictionary<string, NNModel>();


        // Max episodes to run. Only used if > 0
        // Will default to 1 if override models are specified, otherwise 0.
        int m_MaxEpisodes;

        int m_NumSteps;
        int m_PreviousNumSteps;
        int m_PreviousAgentCompletedEpisodes;

        bool m_QuitOnLoadFailure;
        [Tooltip("Debug values to be used in place of the command line for overriding models.")]
        public string debugCommandLineOverride;

        // Static values to keep track of completed episodes and steps across resets
        // These are updated in OnDisable.
        static int s_PreviousAgentCompletedEpisodes;
        static int s_PreviousNumSteps;

        int TotalCompletedEpisodes
        {
            get { return m_PreviousAgentCompletedEpisodes + (m_Agent == null ? 0 : m_Agent.CompletedEpisodes);  }
        }

        int TotalNumSteps
        {
            get { return m_PreviousNumSteps + m_NumSteps; }
        }

        public bool HasOverrides
        {
            get { return m_BehaviorNameOverrides.Count > 0 || !string.IsNullOrEmpty(m_BehaviorNameOverrideDirectory);  }
        }

        public static string GetOverrideBehaviorName(string originalBehaviorName)
        {
            return $"Override_{originalBehaviorName}";
        }

        /// <summary>
        /// Get the asset path to use from the commandline arguments.
        /// </summary>
        /// <returns></returns>
        void GetAssetPathFromCommandLine()
        {
            m_BehaviorNameOverrides.Clear();

            var maxEpisodes = 0;
            string[] commandLineArgsOverride = null;
            if (!string.IsNullOrEmpty(debugCommandLineOverride) && Application.isEditor)
            {
                commandLineArgsOverride = debugCommandLineOverride.Split(' ');
            }

            var args = commandLineArgsOverride ?? Environment.GetCommandLineArgs();
            for (var i = 0; i < args.Length; i++)
            {
                if (args[i] == k_CommandLineModelOverrideFlag && i < args.Length-2)
                {
                    var key = args[i + 1].Trim();
                    var value = args[i + 2].Trim();
                    m_BehaviorNameOverrides[key] = value;
                }
                else if (args[i] == k_CommandLineModelOverrideDirectoryFlag && i < args.Length-1)
                {
                    m_BehaviorNameOverrideDirectory = args[i + 1].Trim();
                }
                else if (args[i] == k_CommandLineQuitAfterEpisodesFlag && i < args.Length-1)
                {
                    Int32.TryParse(args[i + 1], out maxEpisodes);
                }
                else if (args[i] == k_CommandLineQuitOnLoadFailure)
                {
                    m_QuitOnLoadFailure = true;
                }
            }

            if (HasOverrides)
            {
                // If overriding models, set maxEpisodes to 1 or the command line value
                m_MaxEpisodes = maxEpisodes > 0 ? maxEpisodes : 1;
                Debug.Log($"setting m_MaxEpisodes to {maxEpisodes}");
            }
        }

        void OnEnable()
        {
            // Start with these initialized to previous values in the case where we're resetting scenes.
            m_PreviousNumSteps = s_PreviousNumSteps;
            m_PreviousAgentCompletedEpisodes = s_PreviousAgentCompletedEpisodes;

            m_Agent = GetComponent<Agent>();

            GetAssetPathFromCommandLine();
            if (HasOverrides)
            {
                OverrideModel();
            }
        }

        void OnDisable()
        {
            // Update the static episode and step counts.
            // For a single agent in the scene, this will be a straightforward increment.
            // If there are multiple agents, we'll increment the count by the Agent that completed the most episodes.
            s_PreviousAgentCompletedEpisodes = Mathf.Max(s_PreviousAgentCompletedEpisodes, TotalCompletedEpisodes);
            s_PreviousNumSteps = Mathf.Max(s_PreviousNumSteps, TotalNumSteps);
        }

        void FixedUpdate()
        {
            if (m_MaxEpisodes > 0)
            {
                // For Agents without maxSteps, exit as soon as we've hit the target number of episodes.
                // For Agents that specify MaxStep, also make sure we've gone at least that many steps.
                // Since we exit as soon as *any* Agent hits its target, the maxSteps condition keeps us running
                // a bit longer in case there's an early failure.
                if (TotalCompletedEpisodes >= m_MaxEpisodes && TotalNumSteps > m_MaxEpisodes * m_Agent.MaxStep)
                {
                    Debug.Log($"ModelOverride reached {TotalCompletedEpisodes} episodes and {TotalNumSteps} steps. Exiting.");
                    Application.Quit(0);
#if UNITY_EDITOR
                    EditorApplication.isPlaying = false;
#endif
                }
            }
            m_NumSteps++;
        }

        public NNModel GetModelForBehaviorName(string behaviorName)
        {
            if (m_CachedModels.ContainsKey(behaviorName))
            {
                return m_CachedModels[behaviorName];
            }

            string assetPath = null;
            if (m_BehaviorNameOverrides.ContainsKey(behaviorName))
            {
                assetPath = m_BehaviorNameOverrides[behaviorName];
            }
            else if(!string.IsNullOrEmpty(m_BehaviorNameOverrideDirectory))
            {
                assetPath = Path.Combine(m_BehaviorNameOverrideDirectory, $"{behaviorName}.nn");
            }

            if (string.IsNullOrEmpty(assetPath))
            {
                Debug.Log($"No override for BehaviorName {behaviorName}, and no directory set.");
                return null;
            }

            byte[] model = null;
            try
            {
                model = File.ReadAllBytes(assetPath);
            }
            catch(IOException)
            {
                Debug.Log($"Couldn't load file {assetPath} at full path {Path.GetFullPath(assetPath)}", this);
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
            var behaviorName = bp.BehaviorName;

            var nnModel = GetModelForBehaviorName(behaviorName);
            if (nnModel == null && m_QuitOnLoadFailure)
            {
                Debug.Log(
                    $"Didn't find a model for behaviorName {behaviorName}. Make " +
                    $"sure the behaviorName is set correctly in the commandline " +
                    $"and that the model file exists"
                );
                Application.Quit(1);
#if UNITY_EDITOR
                EditorApplication.isPlaying = false;
#endif
            }
            var modelName = nnModel != null ? nnModel.name : "<null>";
            Debug.Log($"Overriding behavior {behaviorName} for agent with model {modelName}");
            // This might give a null model; that's better because we'll fall back to the Heuristic
            m_Agent.SetModel(GetOverrideBehaviorName(behaviorName), nnModel);

        }
    }
}
