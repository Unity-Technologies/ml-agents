using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.IO;
using Unity.Barracuda.ONNX;
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
    /// player.exe --mlagents-override-model-directory /path/to/models
    ///
    /// Additionally, a number of episodes to run can be specified; after this, the application will quit.
    /// Note this will only work with example scenes that have 1:1 Agent:Behaviors. More complicated scenes like WallJump
    /// probably won't override correctly.
    /// </summary>
    public class ModelOverrider : MonoBehaviour
    {
        HashSet<string> k_SupportedExtensions = new HashSet<string> { "nn", "onnx" };
        const string k_CommandLineModelOverrideDirectoryFlag = "--mlagents-override-model-directory";
        const string k_CommandLineModelOverrideExtensionFlag = "--mlagents-override-model-extension";
        const string k_CommandLineQuitAfterEpisodesFlag = "--mlagents-quit-after-episodes";
        const string k_CommandLineQuitAfterSeconds = "--mlagents-quit-after-seconds";
        const string k_CommandLineQuitOnLoadFailure = "--mlagents-quit-on-load-failure";

        // The attached Agent
        Agent m_Agent;

        string m_BehaviorNameOverrideDirectory;

        private List<string> m_OverrideExtensions = new List<string>();

        // Cached loaded NNModels, with the behavior name as the key.
        Dictionary<string, NNModel> m_CachedModels = new Dictionary<string, NNModel>();


        // Max episodes to run. Only used if > 0
        // Will default to 1 if override models are specified, otherwise 0.
        int m_MaxEpisodes;

        // Deadline - exit if the time exceeds this
        DateTime m_Deadline = DateTime.MaxValue;

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
            get { return m_PreviousAgentCompletedEpisodes + (m_Agent == null ? 0 : m_Agent.CompletedEpisodes); }
        }

        int TotalNumSteps
        {
            get { return m_PreviousNumSteps + m_NumSteps; }
        }

        public bool HasOverrides
        {
            get
            {
                GetAssetPathFromCommandLine();
                return !string.IsNullOrEmpty(m_BehaviorNameOverrideDirectory);
            }
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
            var maxEpisodes = 0;
            var timeoutSeconds = 0;

            string[] commandLineArgsOverride = null;
            if (!string.IsNullOrEmpty(debugCommandLineOverride) && Application.isEditor)
            {
                commandLineArgsOverride = debugCommandLineOverride.Split(' ');
            }

            var args = commandLineArgsOverride ?? Environment.GetCommandLineArgs();
            for (var i = 0; i < args.Length; i++)
            {
                if (args[i] == k_CommandLineModelOverrideDirectoryFlag && i < args.Length - 1)
                {
                    m_BehaviorNameOverrideDirectory = args[i + 1].Trim();
                }
                else if (args[i] == k_CommandLineModelOverrideExtensionFlag && i < args.Length - 1)
                {
                    var overrideExtension = args[i + 1].Trim().ToLower();
                    var isKnownExtension = k_SupportedExtensions.Contains(overrideExtension);
                    if (!isKnownExtension)
                    {
                        Debug.LogError($"loading unsupported format: {overrideExtension}");
                        Application.Quit(1);
#if UNITY_EDITOR
                        EditorApplication.isPlaying = false;
#endif
                    }
                    m_OverrideExtensions.Add(overrideExtension);
                }
                else if (args[i] == k_CommandLineQuitAfterEpisodesFlag && i < args.Length - 1)
                {
                    Int32.TryParse(args[i + 1], out maxEpisodes);
                }
                else if (args[i] == k_CommandLineQuitAfterSeconds && i < args.Length - 1)
                {
                    Int32.TryParse(args[i + 1], out timeoutSeconds);
                }
                else if (args[i] == k_CommandLineQuitOnLoadFailure)
                {
                    m_QuitOnLoadFailure = true;
                }
            }

            if (!string.IsNullOrEmpty(m_BehaviorNameOverrideDirectory))
            {
                // If overriding models, set maxEpisodes to 1 or the command line value
                m_MaxEpisodes = maxEpisodes > 0 ? maxEpisodes : 1;
                Debug.Log($"setting m_MaxEpisodes to {maxEpisodes}");
            }

            if (timeoutSeconds > 0)
            {
                m_Deadline = DateTime.Now + TimeSpan.FromSeconds(timeoutSeconds);
                Debug.Log($"setting deadline to {timeoutSeconds} from now.");

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
                else if (DateTime.Now >= m_Deadline)
                {
                    Debug.Log(
                        $"Deadline exceeded. " +
                        $"{TotalCompletedEpisodes}/{m_MaxEpisodes} episodes and " +
                        $"{TotalNumSteps}/{m_MaxEpisodes * m_Agent.MaxStep} steps completed. Exiting.");
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

            if (string.IsNullOrEmpty(m_BehaviorNameOverrideDirectory))
            {
                Debug.Log($"No override directory set.");
                return null;
            }

            // Try the override extensions in order. If they weren't set, try .nn first, then .onnx.
            var overrideExtensions = (m_OverrideExtensions.Count > 0)
                ? m_OverrideExtensions.ToArray()
                : new[] { "nn", "onnx" };

            byte[] rawModel = null;
            bool isOnnx = false;
            string assetName = null;
            foreach (var overrideExtension in overrideExtensions)
            {
                var assetPath = Path.Combine(m_BehaviorNameOverrideDirectory, $"{behaviorName}.{overrideExtension}");
                try
                {
                    rawModel = File.ReadAllBytes(assetPath);
                    isOnnx = overrideExtension.Equals("onnx");
                    assetName = "Override - " + Path.GetFileName(assetPath);
                    break;
                }
                catch (IOException)
                {
                    // Do nothing - try the next extension, or we'll exit if nothing loaded.
                }
            }

            if (rawModel == null)
            {
                Debug.Log($"Couldn't load model file(s) for {behaviorName} in {m_BehaviorNameOverrideDirectory} (full path: {Path.GetFullPath(m_BehaviorNameOverrideDirectory)}");
                // Cache the null so we don't repeatedly try to load a missing file
                m_CachedModels[behaviorName] = null;
                return null;
            }

            var asset = isOnnx ? LoadOnnxModel(rawModel) : LoadBarracudaModel(rawModel);
            asset.name = assetName;
            m_CachedModels[behaviorName] = asset;
            return asset;
        }

        NNModel LoadBarracudaModel(byte[] rawModel)
        {
            var asset = ScriptableObject.CreateInstance<NNModel>();
            asset.modelData = ScriptableObject.CreateInstance<NNModelData>();
            asset.modelData.Value = rawModel;
            return asset;
        }

        NNModel LoadOnnxModel(byte[] rawModel)
        {
            var converter = new ONNXModelConverter(true);
            var onnxModel = converter.Convert(rawModel);

            NNModelData assetData = ScriptableObject.CreateInstance<NNModelData>();
            using (var memoryStream = new MemoryStream())
            using (var writer = new BinaryWriter(memoryStream))
            {
                ModelWriter.Save(writer, onnxModel);
                assetData.Value = memoryStream.ToArray();
            }
            assetData.name = "Data";
            assetData.hideFlags = HideFlags.HideInHierarchy;

            var asset = ScriptableObject.CreateInstance<NNModel>();
            asset.modelData = assetData;
            return asset;
        }


        /// <summary>
        /// Load the NNModel file from the specified path, and give it to the attached agent.
        /// </summary>
        void OverrideModel()
        {
            bool overrideOk = false;
            string overrideError = null;

            m_Agent.LazyInitialize();
            var bp = m_Agent.GetComponent<BehaviorParameters>();
            var behaviorName = bp.BehaviorName;

            NNModel nnModel = null;
            try
            {
                nnModel = GetModelForBehaviorName(behaviorName);
            }
            catch (Exception e)
            {
                overrideError = $"Exception calling GetModelForBehaviorName: {e}";
            }

            if (nnModel == null)
            {
                if (string.IsNullOrEmpty(overrideError))
                {
                    overrideError =
                        $"Didn't find a model for behaviorName {behaviorName}. Make " +
                        "sure the behaviorName is set correctly in the commandline " +
                        "and that the model file exists";
                }
            }
            else
            {
                var modelName = nnModel != null ? nnModel.name : "<null>";
                Debug.Log($"Overriding behavior {behaviorName} for agent with model {modelName}");
                try
                {
                    m_Agent.SetModel(GetOverrideBehaviorName(behaviorName), nnModel);
                    overrideOk = true;
                }
                catch (Exception e)
                {
                    overrideError = $"Exception calling Agent.SetModel: {e}";
                }
            }

            if (!overrideOk && m_QuitOnLoadFailure)
            {
                if (!string.IsNullOrEmpty(overrideError))
                {
                    Debug.LogWarning(overrideError);
                }
                Application.Quit(1);
#if UNITY_EDITOR
                EditorApplication.isPlaying = false;
#endif
            }

        }
    }
}
