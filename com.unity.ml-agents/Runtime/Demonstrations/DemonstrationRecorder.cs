using System.IO.Abstractions;
using System.Text.RegularExpressions;
using UnityEngine;
using System.IO;
using Unity.MLAgents.Policies;
using UnityEngine.Serialization;

namespace Unity.MLAgents.Demonstrations
{
    /// <summary>
    /// The Demonstration Recorder component facilitates the recording of demonstrations
    /// used for imitation learning.
    /// </summary>
    /// <remarks>Add this component to the [GameObject] containing an <see cref="Agent"/>
    /// to enable recording the agent for imitation learning. You must implement the
    /// <see cref="Agent.Heuristic"/> function of the agent to provide manual control
    /// in order to record demonstrations.
    ///
    /// See [Imitation Learning - Recording Demonstrations] for more information.
    ///
    /// [GameObject]: https://docs.unity3d.com/Manual/GameObjects.html
    /// [Imitation Learning - Recording Demonstrations]: https://github.com/Unity-Technologies/ml-agents/blob/release_19_docs/docs//Learning-Environment-Design-Agents.md#recording-demonstrations
    /// </remarks>
    [RequireComponent(typeof(Agent))]
    [AddComponentMenu("ML Agents/Demonstration Recorder", (int)MenuGroup.Default)]
    public class DemonstrationRecorder : MonoBehaviour
    {
        /// <summary>
        /// Whether or not to record demonstrations.
        /// </summary>
        [FormerlySerializedAs("record")]
        [Tooltip("Whether or not to record demonstrations.")]
        public bool Record;

        /// <summary>
        /// Number of steps to record. The editor will stop playing when it reaches this threshold.
        /// Set to zero to record indefinitely.
        /// </summary>
        [Tooltip("Number of steps to record. The editor will stop playing when it reaches this threshold. " +
                 "Set to zero to record indefinitely.")]
        public int NumStepsToRecord;

        /// <summary>
        /// Base demonstration file name. If multiple files are saved, the additional filenames
        /// will have a sequence of unique numbers appended.
        /// </summary>
        [FormerlySerializedAs("demonstrationName")]
        [Tooltip("Base demonstration file name. If multiple files are saved, the additional " +
                 "filenames will have a unique number appended.")]
        public string DemonstrationName;

        /// <summary>
        /// Directory to save the demo files. Will default to a "Demonstrations/" folder in the
        /// Application data path if not specified.
        /// </summary>
        [FormerlySerializedAs("demonstrationDirectory")]
        [Tooltip("Directory to save the demo files. Will default to " +
                 "{Application.dataPath}/Demonstrations if not specified.")]
        public string DemonstrationDirectory;

        DemonstrationWriter m_DemoWriter;
        internal const int MaxNameLength = 16;

        const string k_ExtensionType = ".demo";
        const string k_DefaultDirectoryName = "Demonstrations";
        IFileSystem m_FileSystem;

        Agent m_Agent;

        void OnEnable()
        {
            m_Agent = GetComponent<Agent>();
        }

        void Update()
        {
            if (!Record)
            {
                return;
            }

            LazyInitialize();

            // Quit when num steps to record is reached
            if (NumStepsToRecord > 0 && m_DemoWriter.NumSteps >= NumStepsToRecord)
            {
                Application.Quit(0);
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
#endif
            }
        }

        /// <summary>
        /// Creates demonstration store for use in recording.
        /// Has no effect if the demonstration store was already created.
        /// </summary>
        internal DemonstrationWriter LazyInitialize(IFileSystem fileSystem = null)
        {
            if (m_DemoWriter != null)
            {
                return m_DemoWriter;
            }

            if (m_Agent == null)
            {
                m_Agent = GetComponent<Agent>();
            }

            m_FileSystem = fileSystem ?? new FileSystem();
            var behaviorParams = GetComponent<BehaviorParameters>();
            if (string.IsNullOrEmpty(DemonstrationName))
            {
                DemonstrationName = behaviorParams.BehaviorName;
            }
            if (string.IsNullOrEmpty(DemonstrationDirectory))
            {
                DemonstrationDirectory = Path.Combine(Application.dataPath, k_DefaultDirectoryName);
            }

            DemonstrationName = SanitizeName(DemonstrationName, MaxNameLength);
            var filePath = MakeDemonstrationFilePath(m_FileSystem, DemonstrationDirectory, DemonstrationName);
            var stream = m_FileSystem.File.Create(filePath);
            m_DemoWriter = new DemonstrationWriter(stream);

            AddDemonstrationWriterToAgent(m_DemoWriter);

            return m_DemoWriter;
        }

        /// <summary>
        /// Removes all characters except alphanumerics from demonstration name.
        /// Shorten name if it is longer than the maxNameLength.
        /// </summary>
        internal static string SanitizeName(string demoName, int maxNameLength)
        {
            var rgx = new Regex("[^a-zA-Z0-9 -]");
            demoName = rgx.Replace(demoName, "");
            // If the string is too long, it will overflow the metadata.
            if (demoName.Length > maxNameLength)
            {
                demoName = demoName.Substring(0, maxNameLength);
            }
            return demoName;
        }

        /// <summary>
        /// Gets a unique path for the DemonstrationName in the DemonstrationDirectory.
        /// </summary>
        /// <param name="fileSystem"></param>
        /// <param name="demonstrationDirectory"></param>
        /// <param name="demonstrationName"></param>
        /// <returns></returns>
        internal static string MakeDemonstrationFilePath(
            IFileSystem fileSystem, string demonstrationDirectory, string demonstrationName
        )
        {
            // Create the directory if it doesn't already exist
            if (!fileSystem.Directory.Exists(demonstrationDirectory))
            {
                fileSystem.Directory.CreateDirectory(demonstrationDirectory);
            }

            var literalName = demonstrationName;
            var filePath = Path.Combine(demonstrationDirectory, literalName + k_ExtensionType);
            var uniqueNameCounter = 0;
            while (fileSystem.File.Exists(filePath))
            {
                // TODO should we use a timestamp instead of a counter here? This loops an increasing number of times
                // as the number of demos increases.
                literalName = demonstrationName + "_" + uniqueNameCounter;
                filePath = Path.Combine(demonstrationDirectory, literalName + k_ExtensionType);
                uniqueNameCounter++;
            }

            return filePath;
        }

        /// <summary>
        /// Close the DemonstrationWriter and remove it from the Agent.
        /// Has no effect if the DemonstrationWriter is already closed (or wasn't opened)
        /// </summary>
        public void Close()
        {
            if (m_DemoWriter != null)
            {
                RemoveDemonstrationWriterFromAgent(m_DemoWriter);

                m_DemoWriter.Close();
                m_DemoWriter = null;
            }
        }

        /// <summary>
        /// Clean up the DemonstrationWriter when shutting down or destroying the Agent.
        /// </summary>
        void OnDestroy()
        {
            Close();
        }

        /// <summary>
        /// Add additional DemonstrationWriter to the Agent. It is still up to the user to Close this
        /// DemonstrationWriters when recording is done.
        /// </summary>
        /// <param name="demoWriter"></param>
        public void AddDemonstrationWriterToAgent(DemonstrationWriter demoWriter)
        {
            var behaviorParams = GetComponent<BehaviorParameters>();
            demoWriter.Initialize(
                DemonstrationName,
                behaviorParams.BrainParameters,
                behaviorParams.FullyQualifiedBehaviorName
            );
            m_Agent.DemonstrationWriters.Add(demoWriter);
        }

        /// <summary>
        /// Remove additional DemonstrationWriter to the Agent. It is still up to the user to Close this
        /// DemonstrationWriters when recording is done.
        /// </summary>
        /// <param name="demoWriter"></param>
        public void RemoveDemonstrationWriterFromAgent(DemonstrationWriter demoWriter)
        {
            m_Agent.DemonstrationWriters.Remove(demoWriter);
        }
    }
}
