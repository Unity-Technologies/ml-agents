using System.IO.Abstractions;
using System.Text.RegularExpressions;
using UnityEngine;
using System.IO;
using MLAgents.Policies;

namespace MLAgents.Demonstrations
{
    /// <summary>
    /// Demonstration Recorder Component.
    /// </summary>
    [RequireComponent(typeof(Agent))]
    [AddComponentMenu("ML Agents/Demonstration Recorder", (int)MenuGroup.Default)]
    public class DemonstrationRecorder : MonoBehaviour
    {
        /// <summary>
        /// Whether or not to record demonstrations.
        /// </summary>
        [Tooltip("Whether or not to record demonstrations.")]
        public bool record;

        /// <summary>
        /// Base demonstration file name. If multiple files are saved, the additional filenames
        /// will have a sequence of unique numbers appended.
        /// </summary>
        [Tooltip("Base demonstration file name. If multiple files are saved, the additional " +
                 "filenames will have a unique number appended.")]
        public string demonstrationName;

        /// <summary>
        /// Directory to save the demo files. Will default to a "Demonstrations/" folder in the
        /// Application data path if not specified.
        /// </summary>
        [Tooltip("Directory to save the demo files. Will default to " +
                 "{Application.dataPath}/Demonstrations if not specified.")]
        public string demonstrationDirectory;

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
            if (record)
            {
                LazyInitialize();
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
            if (string.IsNullOrEmpty(demonstrationName))
            {
                demonstrationName = behaviorParams.behaviorName;
            }
            if (string.IsNullOrEmpty(demonstrationDirectory))
            {
                demonstrationDirectory = Path.Combine(Application.dataPath, k_DefaultDirectoryName);
            }

            demonstrationName = SanitizeName(demonstrationName, MaxNameLength);
            var filePath = MakeDemonstrationFilePath(m_FileSystem, demonstrationDirectory, demonstrationName);
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
        /// Gets a unique path for the demonstrationName in the demonstrationDirectory.
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
                demonstrationName,
                behaviorParams.brainParameters,
                behaviorParams.fullyQualifiedBehaviorName
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
