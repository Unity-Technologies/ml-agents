using System.IO.Abstractions;
using System.Text.RegularExpressions;
using UnityEngine;
using System.IO;

namespace MLAgents
{
    /// <summary>
    /// Demonstration Recorder Component.
    /// </summary>
    [RequireComponent(typeof(Agent))]
    [AddComponentMenu("ML Agents/Demonstration Recorder", (int)MenuGroup.Default)]
    public class DemonstrationRecorder : MonoBehaviour
    {
        [Tooltip("Whether or not to record demonstrations.")]
        public bool record;

        [Tooltip("Base demonstration file name. Will have numbers appended to make unique.")]
        public string demonstrationName;

        [Tooltip("Base directory to write the demo files. If null, will use {Application.dataPath}/Demonstrations.")]
        public string demonstrationDirectory;

        DemonstrationStore m_DemoStore;
        internal const int MaxNameLength = 16;

        const string k_ExtensionType = ".demo";
        IFileSystem m_FileSystem;

        void Update()
        {
            if (record)
            {
                InitializeDemoStore();
            }
        }

        /// <summary>
        /// Creates demonstration store for use in recording.
        /// </summary>
        internal DemonstrationStore InitializeDemoStore(IFileSystem fileSystem = null)
        {
            if (m_DemoStore != null)
            {
                return m_DemoStore;
            }

            m_FileSystem = fileSystem ?? new FileSystem();
            var behaviorParams = GetComponent<BehaviorParameters>();
            if (string.IsNullOrEmpty(demonstrationName))
            {
                demonstrationName = behaviorParams.behaviorName;
            }
            if (string.IsNullOrEmpty(demonstrationDirectory))
            {
                demonstrationDirectory = Path.Combine(Application.dataPath, "Demonstrations");
            }

            demonstrationName = SanitizeName(demonstrationName, MaxNameLength);
            var filePath = MakeDemonstrationFilePath(m_FileSystem, demonstrationDirectory, demonstrationName);
            var stream = m_FileSystem.File.Create(filePath);
            m_DemoStore = new DemonstrationStore(stream);

            m_DemoStore.Initialize(
                demonstrationName,
                behaviorParams.brainParameters,
                behaviorParams.fullyQualifiedBehaviorName
            );

            var agent = GetComponent<Agent>();
            if (agent != null)
            {
                agent.DemonstrationStores.Add(m_DemoStore);
            }

            return m_DemoStore;
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
        /// Close the DemonstrationStore and remove it from the Agent.
        /// Has no effect if the DemonstrationStore is already closed (or wasn't opened)
        /// </summary>
        public void Close()
        {
            if (m_DemoStore != null)
            {
                var agent = GetComponent<Agent>();
                if (agent != null)
                {
                    agent.DemonstrationStores.Remove(m_DemoStore);
                }

                m_DemoStore.Close();
                m_DemoStore = null;
            }
        }

        /// <summary>
        /// Clean up the DemonstrationStore when shutting down or destroying the Agent.
        /// </summary>
        void OnDestroy()
        {
            Close();
        }

    }
}
