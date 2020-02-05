using System.IO.Abstractions;
using System.Text.RegularExpressions;
using UnityEngine;
using System.Collections.Generic;
using System.IO;
using MLAgents.Sensor;

namespace MLAgents
{
    /// <summary>
    /// Demonstration Recorder Component.
    /// </summary>
    //[RequireComponent(typeof(BehaviorParameters))]
    [AddComponentMenu("ML Agents/Demonstration Recorder", (int)MenuGroup.Default)]
    public class DemonstrationRecorder : MonoBehaviour
    {
        [Tooltip("Whether or not to record demonstrations.")]
        public bool record;

        [Tooltip("Base demonstration file name. Will have numbers appended to make unique.")]
        public string demonstrationName;

        [Tooltip("Base directory to write the demo files. If null, will use {Application.dataPath}/Demonstrations.")]
        public string demoDirectory;

        DemonstrationStore m_DemoStore;
        public const int MaxNameLength = 16;

        const string k_ExtensionType = ".demo";
        IFileSystem m_FileSystem;

        void Start()
        {
            if (record)
            {
                InitializeDemoStore();
            }
        }

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
        public void InitializeDemoStore(IFileSystem fileSystem = null)
        {
            if (m_DemoStore != null)
            {
                return;
            }

            if (fileSystem != null)
            {
                m_FileSystem = fileSystem;
            }
            else
            {
                m_FileSystem = new FileSystem();
            }

            if (string.IsNullOrEmpty(demoDirectory))
            {
                demoDirectory = Path.Combine(Application.dataPath, "Demonstrations");
            }

            demonstrationName = SanitizeName(demonstrationName, MaxNameLength);
            CreateDirectory();
            var stream = CreateDemonstrationFile(demonstrationName);
            m_DemoStore = new DemonstrationStore(stream);

            var recordingAgent = GetComponent<Agent>();
            var behaviorParams = GetComponent<BehaviorParameters>();
            m_DemoStore.Initialize(
                demonstrationName,
                behaviorParams.brainParameters,
                behaviorParams.fullyQualifiedBehaviorName
            );
            Monitor.Log("Recording Demonstration of Agent: ", recordingAgent?.name);
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
        /// Checks for the existence of the Demonstrations directory
        /// and creates it if it does not exist.
        /// </summary>
        void CreateDirectory()
        {
            if (!m_FileSystem.Directory.Exists(demoDirectory))
            {
                m_FileSystem.Directory.CreateDirectory(demoDirectory);
            }
        }

        /// <summary>
        /// Creates demonstration file and returns a Stream to it.
        /// </summary>
        Stream CreateDemonstrationFile(string demonstrationName)
        {
            // Creates demonstration file.
            var literalName = demonstrationName;
            var filePath = Path.Combine(demoDirectory, literalName + k_ExtensionType);
            var uniqueNameCounter = 0;
            while (m_FileSystem.File.Exists(filePath))
            {
                literalName = demonstrationName + "_" + uniqueNameCounter;
                filePath = Path.Combine(demoDirectory, literalName + k_ExtensionType);
                uniqueNameCounter++;
            }

            return m_FileSystem.File.Create(filePath);
        }

        /// <summary>
        /// Forwards AgentInfo to Demonstration Store.
        /// </summary>
        public void WriteExperience(AgentInfo info, List<ISensor> sensors)
        {
            // TODO remove, all writing goes through the IExperienceWriter
            m_DemoStore.Record(info, sensors);
        }

        public void Close()
        {
            if (m_DemoStore != null)
            {
                m_DemoStore.Close();
                m_DemoStore = null;
            }
        }

        /// <summary>
        /// Closes Demonstration store.
        /// </summary>
        void OnApplicationQuit()
        {
            Close();
        }

        public IExperienceWriter GetExperienceWriter()
        {
            return m_DemoStore;
        }
    }
}
